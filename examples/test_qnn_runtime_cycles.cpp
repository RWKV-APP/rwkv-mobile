#include "c_api.h"
#include <dirent.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
static void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static std::map<std::string,int> descriptors(int cycle) {
 std::map<std::string,int> targets; DIR *dir=opendir("/proc/self/fd"); check(dir,"fd directory");
 const int own=dirfd(dir); char target[4096];
 while(auto *e=readdir(dir)) {
  char *end=nullptr; const long fd=strtol(e->d_name,&end,10); if(end==e->d_name||*end||fd==own)continue;
  const auto p=std::string("/proc/self/fd/")+e->d_name; const auto n=readlink(p.c_str(),target,sizeof(target)-1); check(n>0,"fd target");target[n]=0;++targets[target];
 }
 closedir(dir); int count=0;for(const auto &[name,n]:targets){count+=n;std::printf("RUNTIME_FD_TARGET cycle=%d count=%d target=%s\n",cycle,n,name.c_str());}
 std::printf("RUNTIME_FD_COUNT cycle=%d count=%d\n",cycle,count);std::fflush(stdout);return targets;
}
int main(int argc,char **argv) {
 try {
  check(argc==4,"usage: runtime-cycle model sdk tokenizer");descriptors(-1);std::map<std::string,int> first;
  for(int cycle=0;cycle<3;++cycle) {
   auto runtime=rwkvmobile_runtime_init();check(runtime,"runtime");
   const int model=rwkvmobile_runtime_load_model_with_extra(runtime,argv[1],"qnn",argv[3],argv[2]);check(model>=0,"QNN model");
   check(rwkvmobile_runtime_release(runtime)==0,"runtime release");
   const auto after=descriptors(cycle);if(!cycle)first=after;else check(after==first,"descriptors grow or targets change across fresh Runtime cycles");
  }
  std::puts("QNN_RUNTIME_CYCLES_FD_BOUNDED_PASSED");return 0;
 }catch(const std::exception &e){std::fprintf(stderr,"QNN_RUNTIME_CYCLES_FAILED %s\n",e.what());return 1;}
}
