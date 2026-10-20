// dmalab: PCIe H2D copy vs compute overlap, async-compute queue, host-pointer import (zero copy)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#define CK(x) do{VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"VK err %d at %s:%d\n",r_,__FILE__,__LINE__);exit(1);}}while(0)
static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q0,q1; static uint32_t qf0=0,qf1=0;
static VkCommandPool cp0,cp1; static VkPhysicalDeviceMemoryProperties mp;
static double now(){ return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Buf{VkBuffer b=0; VkDeviceMemory m=0; size_t sz=0; void* map=0;};
static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags f, VkMemoryPropertyFlags avoid=0){
  for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&f)==f&&!(mp.memoryTypes[i].propertyFlags&avoid)) return i;
  fprintf(stderr,"no memtype f=%x\n",f); exit(1);}
static Buf mk(size_t sz, VkBufferUsageFlags u, VkMemoryPropertyFlags f, VkMemoryPropertyFlags avoid=0){
  Buf b; b.sz=sz; VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=sz; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  CK(vkCreateBuffer(dev,&ci,0,&b.b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b.b,&mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=mr.size; ai.memoryTypeIndex=memType(mr.memoryTypeBits,f,avoid);
  CK(vkAllocateMemory(dev,&ai,0,&b.m)); CK(vkBindBufferMemory(dev,b.b,b.m,0)); if(f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) CK(vkMapMemory(dev,b.m,0,VK_WHOLE_SIZE,0,&b.map)); return b;}
static Buf importHost(void* p, size_t sz, VkBufferUsageFlags u){
  Buf b; b.sz=sz; b.map=p;
  VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; ebi.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.pNext=&ebi; ci.size=sz; ci.usage=u; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  CK(vkCreateBuffer(dev,&ci,0,&b.b));
  auto pfn=(PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev,"vkGetMemoryHostPointerPropertiesEXT");
  VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT}; CK(pfn(dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,p,&hp));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b.b,&mr);
  VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT}; imp.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT; imp.pHostPointer=p;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.pNext=&imp; ai.allocationSize=sz; ai.memoryTypeIndex=memType(hp.memoryTypeBits&mr.memoryTypeBits,0);
  CK(vkAllocateMemory(dev,&ai,0,&b.m)); CK(vkBindBufferMemory(dev,b.b,b.m,0)); return b; }
static VkCommandBuffer cbAlloc(VkCommandPool cp){ VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; a.commandPool=cp; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; CK(vkAllocateCommandBuffers(dev,&a,&c)); return c;}
static VkFence fence(){ VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; CK(vkCreateFence(dev,&fi,0,&f)); return f; }
static void submit(VkQueue q, VkCommandBuffer c, VkFence f){ VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; CK(vkQueueSubmit(q,1,&si,f)); }
static void waitf(VkFence f){ CK(vkWaitForFences(dev,1,&f,VK_TRUE,UINT64_MAX)); CK(vkResetFences(dev,1,&f)); }
static std::vector<char> readFile(const char* fn){ FILE* f=fopen(fn,"rb"); if(!f){fprintf(stderr,"open %s\n",fn);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n) exit(1); fclose(f); return v;}

static VkDescriptorSetLayout dsl; static VkPipelineLayout plg; static VkPipeline rdpipe; static VkDescriptorPool dpool;
static VkDescriptorSet mkSet(Buf& a, Buf& o){
  VkDescriptorSetAllocateInfo al{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; al.descriptorPool=dpool; al.descriptorSetCount=1; al.pSetLayouts=&dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&al,&ds));
  VkDescriptorBufferInfo di[2]={{a.b,0,VK_WHOLE_SIZE},{o.b,0,VK_WHOLE_SIZE}}; VkWriteDescriptorSet wr[2];
  for(int i=0;i<2;i++) wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&di[i],0};
  vkUpdateDescriptorSets(dev,2,wr,0,0); return ds; }

int main(int argc,char**argv){
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
  pd=0; for(auto d:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); if(strstr(p.deviceName,"Radeon")) pd=d; }
  vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0); std::vector<VkQueueFamilyProperties> qp(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp.data());
  qf0=0; for(uint32_t i=0;i<nq;i++) if((qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&(qp[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)){qf0=i;break;}
  qf1=qf0; for(uint32_t i=0;i<nq;i++) if(i!=qf0&&(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){qf1=i;break;}
  printf("qf0=%u qf1=%u\n",qf0,qf1);
  float prio=1; VkDeviceQueueCreateInfo qci[2]; for(int i=0;i<2;i++){ qci[i]={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci[i].queueFamilyIndex=i?qf1:qf0; qci[i].queueCount=1; qci[i].pQueuePriorities=&prio; }
  const char* exts[]={"VK_EXT_external_memory_host"};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=(qf1!=qf0)?2:1; dci.pQueueCreateInfos=qci; dci.enabledExtensionCount=1; dci.ppEnabledExtensionNames=exts;
  CK(vkCreateDevice(pd,&dci,0,&dev)); vkGetDeviceQueue(dev,qf0,0,&q0); vkGetDeviceQueue(dev,qf1,0,&q1);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex=qf0; CK(vkCreateCommandPool(dev,&cpi,0,&cp0)); cpi.queueFamilyIndex=qf1; CK(vkCreateCommandPool(dev,&cpi,0,&cp1));
  VkDescriptorSetLayoutBinding bnd[2]; for(int i=0;i<2;i++) bnd[i]={(uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount=2; dli.pBindings=bnd; CK(vkCreateDescriptorSetLayout(dev,&dli,0,&dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,8}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&dsl; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; CK(vkCreatePipelineLayout(dev,&pli,0,&plg));
  { auto code=readFile("streamrd.spv"); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size(); smi.pCode=(const uint32_t*)code.data(); VkShaderModule sm; CK(vkCreateShaderModule(dev,&smi,0,&sm));
    VkComputePipelineCreateInfo c{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; c.layout=plg; c.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",0}; CK(vkCreateComputePipelines(dev,0,1,&c,0,&rdpipe)); }
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,32}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=8; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; CK(vkCreateDescriptorPool(dev,&dpi,0,&dpool));

  const VkBufferUsageFlags U=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  const size_t EXP=1740000; // ~1.66 MiB
  const size_t NEXP=64;
  Buf dbuf=mk(1ull<<30,U,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);      // 1 GiB device (compute stream-read source + copy dst)
  Buf obuf=mk(1<<20,U,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  Buf pin=mk(NEXP*EXP,U,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  memset(pin.map,1,pin.sz);
  Buf dst=mk(NEXP*EXP,U,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  // imported host memory: 256 MiB page-aligned anonymous
  size_t HS=256ull<<20; void* hp=mmap(0,HS,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE,-1,0); memset(hp,2,HS);
  double t=now(); Buf imp=importHost(hp,HS,U); printf("import 256MiB: %.1f us\n",now()-t);
  VkDescriptorSet dsD=mkSet(dbuf,obuf), dsI=mkSet(imp,obuf);
  VkFence f0=fence(), f1=fence();
  auto recCopy=[&](VkCommandPool cp, Buf& src, Buf& d, int nexp){ VkCommandBuffer c=cbAlloc(cp); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    for(int i=0;i<nexp;i++){ VkBufferCopy r{i*EXP,i*EXP,EXP}; vkCmdCopyBuffer(c,src.b,d.b,1,&r);} vkEndCommandBuffer(c); return c; };
  auto recRead=[&](VkDescriptorSet ds, size_t bytes, int reps){ VkCommandBuffer c=cbAlloc(cp0); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,rdpipe); vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,plg,0,1,&ds,0,0);
    uint32_t wgs=2048; uint32_t per=(uint32_t)(bytes/16/wgs); uint32_t pc[2]={per,0}; vkCmdPushConstants(c,plg,VK_SHADER_STAGE_COMPUTE_BIT,0,8,pc);
    for(int r=0;r<reps;r++){ vkCmdDispatch(c,wgs,1,1); VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT}; vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);} vkEndCommandBuffer(c); return c; };
  auto timeIt=[&](const char* name, std::vector<std::pair<VkQueue,VkCommandBuffer>> jobs, double bytes_each_us_div[2]=nullptr){
    // run each job on its queue concurrently, report per-job completion time
    std::vector<double> best; 
    for(int rep=0;rep<5;rep++){
      double s=now(); VkFence fs[2]={f0,f1};
      for(size_t i=0;i<jobs.size();i++) submit(jobs[i].first,jobs[i].second,fs[i]);
      std::vector<double> e(jobs.size());
      for(size_t i=0;i<jobs.size();i++){ waitf(fs[i]); e[i]=now()-s; }
      if(rep==0) best=e; else for(size_t i=0;i<e.size();i++) best[i]=std::min(best[i],e[i]);
    }
    printf("%-44s",name); for(auto b:best) printf(" %9.1f us",b); printf("\n"); return best; };
  size_t cbytes=NEXP*EXP;
  VkCommandBuffer ccopy0=recCopy(cp0,pin,dst,NEXP), ccopy1=recCopy(cp1,pin,dst,NEXP);
  VkCommandBuffer cread=recRead(dsD,1ull<<30,4);   // 4 GiB read
  VkCommandBuffer cpyImp0=recCopy(cp0,imp,dst,NEXP), cpyImp1=recCopy(cp1,imp,dst,NEXP);
  VkCommandBuffer crdImp=recRead(dsI,HS,16);       // 4 GiB read from host memory
  auto r=timeIt("copy pinned->dev (64x1.66MiB) q0",{{q0,ccopy0}}); printf("   => %.2f GB/s\n",cbytes/r[0]/1e3);
  r=timeIt("copy pinned->dev q1(async)",{{q1,ccopy1}}); printf("   => %.2f GB/s\n",cbytes/r[0]/1e3);
  r=timeIt("copy imported-host->dev q0",{{q0,cpyImp0}}); printf("   => %.2f GB/s\n",cbytes/r[0]/1e3);
  r=timeIt("copy imported-host->dev q1",{{q1,cpyImp1}}); printf("   => %.2f GB/s\n",cbytes/r[0]/1e3);
  r=timeIt("compute stream-read 4GiB dev q0",{{q0,cread}}); printf("   => %.1f GB/s\n",4*1073741824.0/r[0]/1e3);
  r=timeIt("compute stream-read 4GiB imported-host q0",{{q0,crdImp}}); printf("   => %.2f GB/s\n",4*268435456.0*4/r[0]/1e3);
  timeIt("OVERLAP: read(q0) + copy pinned(q1)",{{q0,cread},{q1,ccopy1}});
  timeIt("OVERLAP: read(q0) + copy imported(q1)",{{q0,cread},{q1,cpyImp1}});
  // same queue serial: copy then read
  { VkCommandBuffer cs=cbAlloc(cp0); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cs,&bi);
    for(size_t i=0;i<NEXP;i++){ VkBufferCopy rr{i*EXP,i*EXP,EXP}; vkCmdCopyBuffer(cs,pin.b,dst.b,1,&rr);} vkEndCommandBuffer(cs);
    timeIt("SERIAL q0: (read) then copy",{{q0,cread}}); }
  return 0; }
