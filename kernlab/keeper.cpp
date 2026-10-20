#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#define CK(x) do{VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"VK err %d at %d\n",r_,__LINE__);exit(1);}}while(0)
static VkDevice dev; static VkQueue q; static VkCommandPool cp; static VkPhysicalDeviceMemoryProperties mp;
static std::vector<char> readFile(const char* fn){ FILE* f=fopen(fn,"rb"); if(!f){fprintf(stderr,"open %s\n",fn);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n) exit(1); fclose(f); return v;}

int main(int argc,char**argv){
  uint32_t nwg=argc>1?atoi(argv[1]):8, iters=argc>2?atoi(argv[2]):256, secs=argc>3?atoi(argv[3]):5;
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai; VkInstance inst; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
  VkPhysicalDevice pd=0;
  for(auto d:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); if(strstr(p.deviceName,"Radeon")) pd=d; }
  if(!pd) return 1; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=0; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
  CK(vkCreateDevice(pd,&dci,0,&dev)); vkGetDeviceQueue(dev,0,0,&q);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=0; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; CK(vkCreateCommandPool(dev,&cpi,0,&cp));
  size_t sz=256u<<20; uint32_t nvec=sz/16;
  VkBuffer bufs[2]; VkDeviceMemory mem[2];
  for(int i=0;i<2;i++){ VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=i==0?sz:1<<20; ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT; CK(vkCreateBuffer(dev,&ci,0,&bufs[i]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,bufs[i],&mr); uint32_t t=0; for(uint32_t k=0;k<mp.memoryTypeCount;k++) if((mr.memoryTypeBits&(1u<<k))&&(mp.memoryTypes[k].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){t=k;break;}
    VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mi.allocationSize=mr.size; mi.memoryTypeIndex=t; CK(vkAllocateMemory(dev,&mi,0,&mem[i])); CK(vkBindBufferMemory(dev,bufs[i],mem[i],0)); }
  VkDescriptorSetLayoutBinding bnd[2]; for(int i=0;i<2;i++) bnd[i]={(uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount=2; dli.pBindings=bnd; VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dli,0,&dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,16}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&dsl; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev,&pli,0,&pl));
  auto code=readFile("keeper.spv"); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size(); smi.pCode=(const uint32_t*)code.data(); VkShaderModule sm; CK(vkCreateShaderModule(dev,&smi,0,&sm));
  VkPipeline pipe; VkComputePipelineCreateInfo c{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; c.layout=pl; c.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",0}; CK(vkCreateComputePipelines(dev,0,1,&c,0,&pipe));
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpi,0,&dp));
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dp; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsa,&ds));
  VkDescriptorBufferInfo dbi[2]; VkWriteDescriptorSet wr[2]; for(int i=0;i<2;i++){ dbi[i]={bufs[i],0,VK_WHOLE_SIZE}; wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0}; }
  vkUpdateDescriptorSets(dev,2,wr,0,0);
  VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; a.commandPool=cp; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev,&a,&cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cb,&bi);
  vkCmdFillBuffer(cb,bufs[0],0,VK_WHOLE_SIZE,0x01010101);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
  uint32_t pc[4]={nvec,iters,nwg*64+1,0}; vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc); vkCmdDispatch(cb,nwg,1,1);
  vkEndCommandBuffer(cb);
  auto t0=std::chrono::steady_clock::now(); uint64_t cnt=0; double busy=0;
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<secs){
    auto s=std::chrono::steady_clock::now();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb; CK(vkQueueSubmit(q,1,&si,0)); CK(vkQueueWaitIdle(q)); cnt++;
    busy+=std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count(); }
  printf("keeper nwg=%u iters=%u: %llu submits, %.1f us each, %.2f GB/s\n",nwg,iters,(unsigned long long)cnt,busy/cnt*1e6,(double)nwg*64*iters*16*cnt/busy/1e9);
  return 0; }
