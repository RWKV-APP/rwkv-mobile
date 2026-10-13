import CoreAI
import Foundation

// The native runtime serializes each model's inference and state operations.
// Calls run on its worker thread, never on the application's main thread.
private final class Pending<T>: @unchecked Sendable {
    let done = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?
}

private func waitFor<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
    let pending = Pending<T>()
    Task.detached {
        do { pending.result = .success(try await operation()) }
        catch { pending.result = .failure(error) }
        pending.done.signal()
    }
    pending.done.wait()
    return try pending.result!.get()
}

private struct ModelConfig: Decodable {
    let format_version: Int
    let backend: String
    let precision: String
    let model_file: String
    let vocab_size: Int
    let prefill_length: Int
    let prefill_lengths: [Int]?
    let tokenshift_shape: [Int]
    let wkv_shape: [Int]
}

private enum CoreAIError: Error {
    case invalidConfig, missingFunction, invalidOutput, invalidState
}

private func checkedCount(_ shape: [Int]) throws -> Int {
    var count = 1
    for dimension in shape {
        let product = count.multipliedReportingOverflow(by: dimension)
        guard !product.overflow, product.partialValue > 0, product.partialValue <= 100_000_000 else {
            throw CoreAIError.invalidConfig
        }
        count = product.partialValue
    }
    return count
}

@available(macOS 27.0, *)
private func runStep<Scalar: BinaryFloatingPoint & BitwiseCopyable>(_ function: InferenceFunction, input: NDArray,
                     shift: inout NDArray, recurrence: inout NDArray,
                     vocab: Int, readLogits: Bool, as: Scalar.Type) async throws -> [Float] {
    var states = InferenceFunction.MutableViews()
    let scalarType = shift.scalarType
    states.insert(shift.mutableView(as: Scalar.self), for: "tokenshift")
    states.insert(recurrence.mutableView(as: Scalar.self), for: "wkv")
    var outputs = try await function.run(inputs: ["tokens": input], states: states)
    guard readLogits else { return [] }
    guard let value = outputs.remove("logits"), let logits = value.ndArray,
          logits.shape == [1, 1, vocab], logits.scalarType == scalarType
    else { throw CoreAIError.invalidOutput }
    return logits.view(as: Scalar.self).withUnsafePointer { pointer, _, strides in
        (0..<vocab).map { Float(pointer[$0 * strides[2]]) }
    }
}

@available(macOS 27.0, *)
private func stateData<Scalar: BitwiseCopyable>(_ array: NDArray, as: Scalar.Type) -> Data {
    let byteCount = array.shape.reduce(1, *) * MemoryLayout<Scalar>.stride
    return array.view(as: Scalar.self).withUnsafePointer { pointer, _, _ in
        Data(bytes: pointer, count: byteCount)
    }
}

@available(macOS 27.0, *)
private func stateArray<Scalar: BinaryFloatingPoint & BitwiseCopyable>(_ data: Data?, offset: Int,
                                                                    shape: [Int], as: Scalar.Type) -> NDArray {
    var values = [Scalar](repeating: 0, count: shape.reduce(1, *))
    if let data {
        _ = values.withUnsafeMutableBytes { output in
            data.copyBytes(to: output, from: offset..<(offset + output.count))
        }
    }
    return NDArray(scalars: values, shape: shape)
}

@available(macOS 27.0, *)
private final class ModelContext: @unchecked Sendable {
    let model: AIModel
    let config: ModelConfig
    let decode: InferenceFunction
    let prefills: [(length: Int, function: InferenceFunction)]
    var tokenshift: NDArray
    var wkv: NDArray

    init(directory: String) async throws {
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        config = try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        guard config.format_version == 1, config.backend == "coreai",
              ["float16", "float32"].contains(config.precision), config.model_file == "rwkv.aimodel",
              config.vocab_size > 0, config.vocab_size <= 1_000_000,
              config.prefill_length > 1, config.prefill_length <= 4096,
              config.tokenshift_shape.count == 3, config.wkv_shape.count == 4,
              (config.tokenshift_shape + config.wkv_shape).allSatisfy({ $0 > 0 && $0 <= 65536 })
        else { throw CoreAIError.invalidConfig }
        let shiftCount = try checkedCount(config.tokenshift_shape)
        let wkvCount = try checkedCount(config.wkv_shape)
        if config.precision == "float16" {
            tokenshift = NDArray(scalars: [Float16](repeating: 0, count: shiftCount), shape: config.tokenshift_shape)
            wkv = NDArray(scalars: [Float16](repeating: 0, count: wkvCount), shape: config.wkv_shape)
        } else {
            tokenshift = NDArray(scalars: [Float](repeating: 0, count: shiftCount), shape: config.tokenshift_shape)
            wkv = NDArray(scalars: [Float](repeating: 0, count: wkvCount), shape: config.wkv_shape)
        }
        // GPU is a preference, not evidence that every operation ran on GPU.
        model = try await AIModel(contentsOf: root.appendingPathComponent(config.model_file), options: .init(preferredComputeUnitKind: .gpu))
        guard let decode = try model.loadFunction(named: "decode") else { throw CoreAIError.missingFunction }
        let lengths = config.prefill_lengths ?? [config.prefill_length]
        guard !lengths.isEmpty, Set(lengths).count == lengths.count,
              lengths.max() == config.prefill_length,
              lengths.allSatisfy({ $0 > 1 && $0 <= 4096 }) else { throw CoreAIError.invalidConfig }
        var loadedPrefills: [(length: Int, function: InferenceFunction)] = []
        for length in lengths.sorted(by: >) {
            let name = length == config.prefill_length ? "prefill" : "prefill_\(length)"
            guard let function = try model.loadFunction(named: name) else { throw CoreAIError.missingFunction }
            loadedPrefills.append((length, function))
        }
        self.decode = decode
        self.prefills = loadedPrefills
        for function in [decode] + loadedPrefills.map({ $0.function }) {
            guard case .ndArray(let shiftDescriptor) = function.descriptor.stateDescriptor(of: "tokenshift"),
                  case .ndArray(let wkvDescriptor) = function.descriptor.stateDescriptor(of: "wkv"),
                  shiftDescriptor.shape == config.tokenshift_shape,
                  wkvDescriptor.shape == config.wkv_shape,
                  shiftDescriptor.scalarType == tokenshift.scalarType, wkvDescriptor.scalarType == wkv.scalarType
            else { throw CoreAIError.invalidState }
        }
    }

    func evaluate(_ tokens: [Int32]) async throws -> [Float] {
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 && Int($0) < config.vocab_size }) else {
            throw CoreAIError.invalidConfig
        }
        var offset = 0
        var finalLogits: [Float] = []
        while offset < tokens.count {
            let selected = prefills.first { $0.length <= tokens.count - offset }
            let length = selected?.length ?? 1
            let input = NDArray(scalars: tokens[offset..<(offset + length)], shape: [1, length])
            let function = selected?.function ?? decode
            if config.precision == "float16" {
                finalLogits = try await runStep(function, input: input, shift: &tokenshift, recurrence: &wkv,
                                              vocab: config.vocab_size, readLogits: offset + length == tokens.count, as: Float16.self)
            } else {
                finalLogits = try await runStep(function, input: input, shift: &tokenshift, recurrence: &wkv,
                                              vocab: config.vocab_size, readLogits: offset + length == tokens.count, as: Float.self)
            }
            offset += length
        }
        return finalLogits
    }

    var stateCount: Int { config.tokenshift_shape.reduce(1, *) + config.wkv_shape.reduce(1, *) }
    var scalarBytes: Int { config.precision == "float16" ? 2 : 4 }
    var stateBytes: Int { stateCount * scalarBytes }

    func snapshot() -> Data {
        // Both state arrays are allocated here from contiguous scalar sequences.
        if config.precision == "float16" {
            return stateData(tokenshift, as: Float16.self) + stateData(wkv, as: Float16.self)
        }
        return stateData(tokenshift, as: Float.self) + stateData(wkv, as: Float.self)
    }

    func restore(_ data: Data?) throws {
        guard data == nil || data!.count == stateBytes else { throw CoreAIError.invalidState }
        let split = config.tokenshift_shape.reduce(1, *) * scalarBytes
        if config.precision == "float16" {
            tokenshift = stateArray(data, offset: 0, shape: config.tokenshift_shape, as: Float16.self)
            wkv = stateArray(data, offset: split, shape: config.wkv_shape, as: Float16.self)
        } else {
            tokenshift = stateArray(data, offset: 0, shape: config.tokenshift_shape, as: Float.self)
            wkv = stateArray(data, offset: split, shape: config.wkv_shape, as: Float.self)
        }
    }
}

private func logError(_ error: Error) {
    let message = "Core AI: \(error)\n"
    _ = message.withCString { fputs($0, stderr) }
}

@_cdecl("rwkv_coreai_available")
public func coreAIAvailable() -> Int32 {
    if #available(macOS 27.0, *) { return 1 }
    return 0
}

@_cdecl("rwkv_coreai_load")
public func coreAILoad(_ path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard #available(macOS 27.0, *), let path else { return nil }
    let directory = String(cString: path)
    do {
        let context = try waitFor { try await ModelContext(directory: directory) }
        return Unmanaged.passRetained(context).toOpaque()
    } catch { logError(error); return nil }
}

@_cdecl("rwkv_coreai_free")
public func coreAIFree(_ handle: UnsafeMutableRawPointer?) {
    guard #available(macOS 27.0, *), let handle else { return }
    Unmanaged<ModelContext>.fromOpaque(handle).release()
}

@_cdecl("rwkv_coreai_eval")
public func coreAIEval(_ handle: UnsafeMutableRawPointer?, _ tokens: UnsafePointer<Int32>?, _ count: Int32, _ logits: UnsafeMutablePointer<Float>?, _ capacity: Int32) -> Int32 {
    guard #available(macOS 27.0, *), let handle, let tokens, let logits, count > 0 else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard capacity == context.config.vocab_size else { return -1 }
    let input = Array(UnsafeBufferPointer(start: tokens, count: Int(count)))
    do {
        let result = try waitFor { try await context.evaluate(input) }
        guard result.count == Int(capacity), result.allSatisfy({ $0.isFinite }) else { return -1 }
        logits.update(from: result, count: result.count)
        return 0
    } catch { logError(error); return -1 }
}

@_cdecl("rwkv_coreai_state_bytes")
public func coreAIStateBytes(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    guard #available(macOS 27.0, *), let handle else { return -1 }
    return Int64(Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue().stateBytes)
}

@_cdecl("rwkv_coreai_read_state")
public func coreAIReadState(_ handle: UnsafeMutableRawPointer?, _ output: UnsafeMutableRawPointer?, _ bytes: Int64) -> Int32 {
    guard #available(macOS 27.0, *), let handle, let output else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard bytes == context.stateBytes else { return -1 }
    context.snapshot().withUnsafeBytes { data in output.copyMemory(from: data.baseAddress!, byteCount: data.count) }
    return 0
}

@_cdecl("rwkv_coreai_write_state")
public func coreAIWriteState(_ handle: UnsafeMutableRawPointer?, _ input: UnsafeRawPointer?, _ bytes: Int64) -> Int32 {
    guard #available(macOS 27.0, *), let handle else { return -1 }
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard bytes == context.stateBytes else { return -1 }
    let data = input.map { Data(bytes: $0, count: Int(bytes)) }
    do { try context.restore(data); return 0 }
    catch { logError(error); return -1 }
}
