// kernlab: standalone raw-Vulkan matvec kernel lab (repacked 16B-lane layouts)
// usage: kernlab <spv> <rows> <K> <rows_per_wg> <pool> [mode]
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#define CK(x) do{VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"VK err %d at %s:%d\n",r_,__FILE__,__LINE__);exit(1);}}while(0)
static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q; static uint32_t qf;
static VkCommandPool cp; static VkPhysicalDeviceMemoryProperties mp; static float tsPeriod;
struct Buf{VkBuffer b=0; VkDeviceMemory m=0; size_t sz=0;};
static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f){
  for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&f)==f) return i;
  fprintf(stderr,"no memtype\n"); exit(1);}
static Buf mk(size_t sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f){
  Buf b; b.sz=sz; VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  CK(vkCreateBuffer(dev,&ci,0,&b.b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b.b,&mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=mr.size; ai.memoryTypeIndex=memType(mr.memoryTypeBits,f);
  CK(vkAllocateMemory(dev,&ai,0,&b.m)); CK(vkBindBufferMemory(dev,b.b,b.m,0)); return b;}
static VkCommandBuffer cbAlloc(){ VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; a.commandPool=cp; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; CK(vkAllocateCommandBuffers(dev,&a,&c)); return c;}
static void submitWait(VkCommandBuffer c){ VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; CK(vkQueueSubmit(q,1,&si,0)); CK(vkQueueWaitIdle(q)); }
static Buf devBuf(const void* data, size_t sz){
  Buf d=mk(sz,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  const size_t CH=64<<20; Buf s=mk(std::min(sz,CH),VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  void* p; CK(vkMapMemory(dev,s.m,0,VK_WHOLE_SIZE,0,&p));
  for(size_t off=0;off<sz;off+=CH){ size_t n=std::min(CH,sz-off); memcpy(p,(const char*)data+off,n);
    VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    VkBufferCopy r{0,off,n}; vkCmdCopyBuffer(c,s.b,d.b,1,&r); vkEndCommandBuffer(c); submitWait(c); }
  vkUnmapMemory(dev,s.m); vkDestroyBuffer(dev,s.b,0); vkFreeMemory(dev,s.m,0); return d;}
static std::vector<char> readFile(const char* fn){ FILE* f=fopen(fn,"rb"); if(!f){fprintf(stderr,"open %s\n",fn);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n) exit(1); fclose(f); return v;}
static uint64_t rng=88172645463325252ull; static uint32_t rnd(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>16);}
static float rndf(){ return (rnd()&0xffff)/32768.0f-1.0f; }
static uint16_t f2h(float f){ _Float16 h=(_Float16)f; uint16_t u; memcpy(&u,&h,2); return u;}
static float h2f(uint16_t u){ _Float16 h; memcpy(&h,&u,2); return (float)h;}


struct Pipe{VkPipeline p=0; VkPipelineLayout pl=0; VkDescriptorSet ds=0;};
static VkDescriptorSetLayout dsl; static VkPipelineLayout plg; static VkDescriptorPool dpool;
static VkPipeline mkPipe(const char* spv, uint32_t sg){
  auto code=readFile(spv); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size(); smi.pCode=(const uint32_t*)code.data(); VkShaderModule sm; CK(vkCreateShaderModule(dev,&smi,0,&sm));
  VkSpecializationMapEntry me{0,0,4}; VkSpecializationInfo si{1,&me,4,&sg};
  VkComputePipelineCreateInfo c{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; c.layout=plg; c.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",&si};
  VkPipeline p; CK(vkCreateComputePipelines(dev,0,1,&c,0,&p)); return p; }
static VkDescriptorSet mkSet(Buf** bs,int n){
  VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; a.descriptorPool=dpool; a.descriptorSetCount=1; a.pSetLayouts=&dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&a,&ds));
  VkDescriptorBufferInfo dbi[5]; VkWriteDescriptorSet wr[5];
  for(int i=0;i<n;i++){ dbi[i]={bs[i]->b,0,VK_WHOLE_SIZE}; wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0}; }
  vkUpdateDescriptorSets(dev,n,wr,0,0); return ds; }
static void bar(VkCommandBuffer c){ VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
  vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0); }
static uint16_t bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

// hclab A.spv|- B.spv|- [pool] [ref.bin]   env: HC_AG="x,y,z" HC_BG="x,y,z"
int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: hclab A.spv|- B.spv|- [pool] [ref.bin]\n");return 1;}
  const char* sa=argv[1]; const char* sb=argv[2]; uint32_t P=argc>3?atoi(argv[3]):48; const char* refn=argc>4?argv[4]:0;
  uint32_t ag[3]={81,4,1}, bg[3]={320,1,1};
  if(getenv("HC_AG")) sscanf(getenv("HC_AG"),"%u,%u,%u",&ag[0],&ag[1],&ag[2]);
  if(getenv("HC_BG")) sscanf(getenv("HC_BG"),"%u,%u,%u",&bg[0],&bg[1],&bg[2]);
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
  pd=0; for(auto d:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); if(strstr(p.deviceName,"Radeon")){pd=d; tsPeriod=p.limits.timestampPeriod;} }
  if(!pd){fprintf(stderr,"no radeon\n");return 1;}
  vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0); std::vector<VkQueueFamilyProperties> qp(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp.data());
  qf=0; for(uint32_t i=0;i<nq;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.shaderFloat16=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE; f12.shaderInt8=VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.storageBuffer16BitAccess=VK_TRUE; f11.pNext=&f12;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.features.shaderInt16=VK_TRUE; f2.pNext=&f11;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
  CK(vkCreateDevice(pd,&dci,0,&dev)); vkGetDeviceQueue(dev,qf,0,&q);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=qf; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; CK(vkCreateCommandPool(dev,&cpi,0,&cp));

  const uint32_t NE=2560, LO=320, HC=4;
  const size_t wd_n=(size_t)LO*HC*NE, wi_n=(size_t)HC*HC*NE, wu_n=(size_t)HC*NE*LO; // bf16 elements per slot
  std::vector<uint16_t> wd(wd_n), wi(wi_n), wu(wu_n);
  for(auto&v:wd) v=bf(rndf()*0.04f); for(auto&v:wi) v=bf(rndf()*0.04f); for(auto&v:wu) v=bf(rndf()*0.04f);
  std::vector<uint16_t> wdP(wd_n*P), wiP(wi_n*P), wuP(wu_n*P);
  for(uint32_t i=0;i<P;i++){ memcpy(&wdP[i*wd_n],wd.data(),wd_n*2); memcpy(&wiP[i*wi_n],wi.data(),wi_n*2); memcpy(&wuP[i*wu_n],wu.data(),wu_n*2); }
  std::vector<float> x(HC*NE), g(HC*NE); for(auto&v:x) v=rndf()*3.0f; for(auto&v:g) v=0.5f+rndf()*0.3f;
  std::vector<float> zeros(16384,0.f);
  Buf bx=devBuf(x.data(),x.size()*4), bg_=devBuf(g.data(),g.size()*4), bwd=devBuf(wdP.data(),wdP.size()*2), bwi=devBuf(wiP.data(),wiP.size()*2), bwu=devBuf(wuP.data(),wuP.size()*2), bo=devBuf(zeros.data(),zeros.size()*4);
  VkDescriptorSetLayoutBinding bnd[5]; for(int i=0;i<5;i++) bnd[i]={(uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount=5; dli.pBindings=bnd; CK(vkCreateDescriptorSetLayout(dev,&dli,0,&dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,44}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&dsl; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; CK(vkCreatePipelineLayout(dev,&pli,0,&plg));
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,32}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=4; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; CK(vkCreateDescriptorPool(dev,&dpi,0,&dpool));
  bool hasA=strcmp(sa,"-"), hasB=strcmp(sb,"-");
  VkPipeline pa=hasA?mkPipe(sa,64):0, pb=hasB?mkPipe(sb,64):0;
  Buf* abs_[5]={&bx,&bg_,&bwd,&bwi,&bo}; Buf* bbs[4]={&bx,&bg_,&bwu,&bo};
  VkDescriptorSet dsa=mkSet(abs_,5), dsb=mkSet(bbs,4);
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType=VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount=2; VkQueryPool qpool; CK(vkCreateQueryPool(dev,&qpi,0,&qpool));
  auto record=[&](VkCommandBuffer c,int i){
    uint32_t sl=i%P; uint32_t pc[11]={NE,LO,0,0,(uint32_t)(sl*(wd_n/2)),(uint32_t)(sl*(wi_n/2)),(uint32_t)(sl*(wu_n/2)),0,0,0,4};
    float eps=1e-6f, ih=0.25f; memcpy(&pc[8],&eps,4); memcpy(&pc[9],&ih,4);
    vkCmdPushConstants(c,plg,VK_SHADER_STAGE_COMPUTE_BIT,0,44,pc);
    if(hasA){ vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pa); vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,plg,0,1,&dsa,0,0); vkCmdDispatch(c,ag[0],ag[1],ag[2]); bar(c); }
    if(hasB){ vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pb); vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,plg,0,1,&dsb,0,0); vkCmdDispatch(c,bg[0],bg[1],bg[2]); bar(c); } };
  auto run=[&](int iters){
    VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    vkCmdResetQueryPool(c,qpool,0,2); vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qpool,0);
    for(int i=0;i<iters;i++) record(c,i);
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qpool,1); vkEndCommandBuffer(c); submitWait(c);
    uint64_t t[2]; CK(vkGetQueryPoolResults(dev,qpool,0,2,16,t,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)); vkFreeCommandBuffers(dev,cp,1,&c);
    return (double)(t[1]-t[0])*tsPeriod*1e-3/iters; };
  int NI=std::max<int>(P*2,96); run(NI); std::vector<double> ts; for(int r=0;r<9;r++) ts.push_back(run(NI)); std::sort(ts.begin(),ts.end());
  printf("%s %s pool=%u: min %.2f med %.2f us/call\n",sa,sb,P,ts[0],ts[4]);
  // correctness: single call on slot 0 -> first NE+HC+LO+HC floats
  { VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi); record(c,0); vkEndCommandBuffer(c); submitWait(c); }
  const size_t NO=NE+HC+LO+HC; std::vector<float> out(NO);
  { Buf s=mk(NO*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi); VkBufferCopy r{0,0,NO*4}; vkCmdCopyBuffer(c,bo.b,s.b,1,&r); vkEndCommandBuffer(c); submitWait(c);
    void* p; vkMapMemory(dev,s.m,0,VK_WHOLE_SIZE,0,&p); memcpy(out.data(),p,NO*4); }
  if(refn && hasA && hasB){ FILE* f=fopen(refn,"rb"); if(!f){ f=fopen(refn,"wb"); fwrite(out.data(),4,NO,f); fclose(f); printf("wrote ref %s\n",refn);} else { std::vector<float> ref(NO); if(fread(ref.data(),4,NO,f)!=NO) return 1; fclose(f);
      double me=0,mr=0; for(size_t i=0;i<NO;i++){ me=std::max(me,(double)fabsf(out[i]-ref[i])); mr=std::max(mr,(double)fabsf(ref[i])); } printf("verify vs ref: max|err|=%.3g max|ref|=%.3g\n",me,mr); } }
  return 0; }
