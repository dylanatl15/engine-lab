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

int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage\n");return 1;}
  const char* spv=argv[1]; uint32_t rows=atoi(argv[2]), K=atoi(argv[3]), rpw=atoi(argv[4]), P=atoi(argv[5]);
  uint32_t nsb=K/32;
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
  pd=0; for(auto d:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); if(strstr(p.deviceName,"Radeon")){pd=d; tsPeriod=p.limits.timestampPeriod; printf("dev %s\n",p.deviceName);} }
  if(!pd){fprintf(stderr,"no radeon\n");return 1;}
  vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0); std::vector<VkQueueFamilyProperties> qp(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp.data());
  qf=0; for(uint32_t i=0;i<nq;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.shaderFloat16=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE; f12.uniformAndStorageBuffer8BitAccess=VK_TRUE; f12.shaderInt8=VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.storageBuffer16BitAccess=VK_TRUE; f11.uniformAndStorageBuffer16BitAccess=VK_TRUE; f11.pNext=&f12;
  VkPhysicalDeviceShaderIntegerDotProductFeatures fdp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES}; fdp.shaderIntegerDotProduct=VK_TRUE; fdp.pNext=&f11;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.features.shaderInt16=VK_TRUE; f2.features.shaderInt64=VK_FALSE; f2.pNext=&fdp;
  const char* exts[]={"VK_KHR_shader_integer_dot_product","VK_KHR_shader_float16_int8","VK_KHR_16bit_storage","VK_KHR_8bit_storage"};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=4; dci.ppEnabledExtensionNames=exts;
  CK(vkCreateDevice(pd,&dci,0,&dev)); vkGetDeviceQueue(dev,qf,0,&q);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=qf; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; CK(vkCreateCommandPool(dev,&cpi,0,&cp));

  // ---- data: 4-bit linear repacked weights. per (row,sb): 16B qs, fp16 dw. x quantized q8 per 32.
  size_t nqs=(size_t)rows*nsb;               // uvec4 count
  std::vector<uint8_t> qs(nqs*16); for(auto& b:qs) b=rnd()&0xff;
  std::vector<uint16_t> dw(nqs); for(auto& d:dw) d=f2h((rnd()&0xff)/256.0f*0.02f+0.001f);
  std::vector<float> x(K); for(auto& v:x) v=rndf();
  static const int8_t kv[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
  bool iq4=getenv("KL_IQ4")!=0, xperm=getenv("KL_XPERM")!=0;
  std::vector<int8_t> xq(K); std::vector<float> xm(2*nsb);
  for(uint32_t s=0;s<nsb;s++){ float am=0; for(int j=0;j<32;j++) am=std::max(am,fabsf(x[s*32+j])); float d=am/127.f, id=d?1/d:0; int sum=0;
    for(int j=0;j<32;j++){ int v=(int)roundf(x[s*32+j]*id); xq[s*32+j]=v; sum+=v; } xm[2*s]=d; xm[2*s+1]=iq4?0.0f:8.0f*d*sum; }
  std::vector<int8_t> xdev=xq; if(xperm) for(uint32_t s=0;s<nsb;s++) for(int i=0;i<8;i++){ xdev[s*32+4*i]=xq[s*32+2*i]; xdev[s*32+4*i+1]=xq[s*32+2*i+16]; xdev[s*32+4*i+2]=xq[s*32+2*i+1]; xdev[s*32+4*i+3]=xq[s*32+2*i+17]; }
  // pool replicate
  std::vector<uint8_t> qsP(qs.size()*P); for(uint32_t i=0;i<P;i++) memcpy(&qsP[i*qs.size()],qs.data(),qs.size());
  std::vector<uint16_t> dwP(dw.size()*P); for(uint32_t i=0;i<P;i++) memcpy(&dwP[i*dw.size()],dw.data(),dw.size()*2);
  bool real=getenv("KL_REAL")!=0; size_t copysz=((nqs*17+127)/128)*128;
  std::vector<uint8_t> realP; if(real){ realP.assign(copysz*P,0); for(auto& b:realP) b=rnd()&0xff; }
  std::vector<uint8_t> xreal; if(real){ xreal.assign((size_t)(K/128+1)*144,0); for(auto& b:xreal) b=rnd()&0x3f; for(size_t i=0;i<xreal.size();i+=144){ for(int h=0;h<8;h++) xreal[i+h*2+1]=0x30; } }
  Buf bq=devBuf(real?realP.data():qsP.data(),real?realP.size():qsP.size()), bd=devBuf(dwP.data(),dwP.size()*2), bx=real?devBuf(xreal.data(),xreal.size()):devBuf(xdev.data(),K), bm=devBuf(xm.data(),xm.size()*4);
  std::vector<float> zeros(rows,0.f); Buf bo=devBuf(zeros.data(),rows*4);
  // pipeline
  auto code=readFile(spv); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size(); smi.pCode=(const uint32_t*)code.data(); VkShaderModule sm; CK(vkCreateShaderModule(dev,&smi,0,&sm));
  VkDescriptorSetLayoutBinding bnd[5]; for(int i=0;i<5;i++){bnd[i]={ (uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};}
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount=5; dli.pBindings=bnd; VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dli,0,&dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,16}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&dsl; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev,&pli,0,&pl));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.layout=pl; cpci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",0}; VkPipeline pipe; CK(vkCreateComputePipelines(dev,0,1,&cpci,0,&pipe));
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpi,0,&dp));
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dp; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsa,&ds));
  Buf* bufs[5]={&bq,&bd,&bx,&bm,&bo}; VkDescriptorBufferInfo dbi[5]; VkWriteDescriptorSet wr[5];
  for(int i=0;i<5;i++){ dbi[i]={bufs[i]->b,0,VK_WHOLE_SIZE}; wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0}; }
  vkUpdateDescriptorSets(dev,5,wr,0,0);
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType=VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount=2; VkQueryPool qpool; CK(vkCreateQueryPool(dev,&qpi,0,&qpool));
  uint32_t groups=rows/rpw; int NI=std::max<int>(P*4,64);
  auto run=[&](int iters){
    VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    vkCmdResetQueryPool(c,qpool,0,2); vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qpool,0);
    for(int i=0;i<iters;i++){ uint32_t pc[4]={nsb,rows,(uint32_t)((i%P)*nqs),(uint32_t)((i%P)*nqs)}; if(real){ pc[2]=(uint32_t)((i%P)*copysz/16); pc[3]=(uint32_t)(((i%P)*copysz+nqs*16)/8); } vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc); vkCmdDispatch(c,groups,1,1);
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
      vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0); }
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qpool,1); vkEndCommandBuffer(c); submitWait(c);
    uint64_t t[2]; CK(vkGetQueryPoolResults(dev,qpool,0,2,16,t,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)); vkFreeCommandBuffers(dev,cp,1,&c);
    return (double)(t[1]-t[0])*tsPeriod*1e-3/iters; };
  run(NI); std::vector<double> ts; for(int r=0;r<7;r++) ts.push_back(run(NI)); std::sort(ts.begin(),ts.end());
  double bytes=real?(double)nqs*17:(double)nqs*18; // 16B qs + 2B dw
  printf("rows=%u K=%u pool=%u: min %.2f us  med %.2f us  -> %.1f GB/s (min) %.1f GB/s (med)  [%.2f MB]\n",rows,K,P,ts[0],ts[3],bytes/ts[0]*1e-3,bytes/ts[3]*1e-3,bytes/1e6);
  // verify (output written by last dispatch)
  std::vector<float> out(rows); { Buf s=mk(rows*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer c=cbAlloc(); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi); VkBufferCopy r{0,0,rows*4}; vkCmdCopyBuffer(c,bo.b,s.b,1,&r); vkEndCommandBuffer(c); submitWait(c);
    void* p; vkMapMemory(dev,s.m,0,VK_WHOLE_SIZE,0,&p); memcpy(out.data(),p,rows*4); }
  double maxerr=0,maxref=0; int bad=0;
  for(uint32_t r=0;r<rows;r++){ double acc=0; for(uint32_t s=0;s<nsb;s++){ const uint8_t* b=&qs[((size_t)r*nsb+s)*16]; int dot=0;
      for(int j=0;j<16;j++){ if(iq4){dot+=kv[b[j]&15]*xq[s*32+j]; dot+=kv[b[j]>>4]*xq[s*32+16+j];} else {dot+=(b[j]&15)*xq[s*32+j]; dot+=(b[j]>>4)*xq[s*32+16+j];} }
      acc+=(double)h2f(dw[(size_t)r*nsb+s])*((double)xm[2*s]*dot-xm[2*s+1]); }
    double e=fabs(acc-out[r]); maxerr=std::max(maxerr,e); maxref=std::max(maxref,fabs(acc)); if(e>1e-3*std::max(1.0,fabs(acc))) bad++; }
  printf("verify: max|err|=%.3g max|ref|=%.3g bad=%d/%u\n",maxerr,maxref,bad,rows);
  return 0; }
