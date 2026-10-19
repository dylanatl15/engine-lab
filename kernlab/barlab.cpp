#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
#define CK(x) do{VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"VK err %d at %d\n",r_,__LINE__);exit(1);}}while(0)
static VkDevice dev; static VkQueue q; static VkCommandPool cp; static VkPhysicalDeviceMemoryProperties mp;
static std::vector<char> readFile(const char* fn){ FILE* f=fopen(fn,"rb"); if(!f){fprintf(stderr,"open %s\n",fn);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> v(n); if(fread(v.data(),1,n,f)!=(size_t)n) exit(1); fclose(f); return v;}
int main(int argc,char**argv){
  uint32_t nwg=argc>1?atoi(argv[1]):240, nph=argc>2?atoi(argv[2]):100;
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai; VkInstance inst; CK(vkCreateInstance(&ici,0,&inst));
  uint32_t n=0; vkEnumeratePhysicalDevices(inst,&n,0); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst,&n,pds.data());
  VkPhysicalDevice pd=0; float tsp=1;
  for(auto d:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); if(strstr(p.deviceName,"Radeon")){pd=d; tsp=p.limits.timestampPeriod;} }
  if(!pd) return 1; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=0; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
  CK(vkCreateDevice(pd,&dci,0,&dev)); vkGetDeviceQueue(dev,0,0,&q);
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=0; cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; CK(vkCreateCommandPool(dev,&cpi,0,&cp));
  uint32_t total=nwg*64; size_t sz=std::max<size_t>(total*4,4096);
  VkBuffer bufs[3]; VkDeviceMemory mem[3];
  for(int i=0;i<3;i++){ VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=i==2?4096:sz; ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT; CK(vkCreateBuffer(dev,&ci,0,&bufs[i]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,bufs[i],&mr); uint32_t t=0; for(uint32_t k=0;k<mp.memoryTypeCount;k++) if((mr.memoryTypeBits&(1u<<k))&&(mp.memoryTypes[k].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){t=k;break;}
    VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mi.allocationSize=mr.size; mi.memoryTypeIndex=t; CK(vkAllocateMemory(dev,&mi,0,&mem[i])); CK(vkBindBufferMemory(dev,bufs[i],mem[i],0)); }
  VkDescriptorSetLayoutBinding bnd[3]; for(int i=0;i<3;i++) bnd[i]={(uint32_t)i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount=3; dli.pBindings=bnd; VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev,&dli,0,&dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,16}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount=1; pli.pSetLayouts=&dsl; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev,&pli,0,&pl));
  VkPipeline pipes[2]; const char* fn[2]={"bar_chain.spv","bar_pers.spv"};
  for(int i=0;i<2;i++){ auto code=readFile(fn[i]); VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size(); smi.pCode=(const uint32_t*)code.data(); VkShaderModule sm; CK(vkCreateShaderModule(dev,&smi,0,&sm));
    VkComputePipelineCreateInfo c{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; c.layout=pl; c.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,0,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",0}; CK(vkCreateComputePipelines(dev,0,1,&c,0,&pipes[i])); }
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev,&dpi,0,&dp));
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dp; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev,&dsa,&ds));
  VkDescriptorBufferInfo dbi[3]; VkWriteDescriptorSet wr[3]; for(int i=0;i<3;i++){ dbi[i]={bufs[i],0,VK_WHOLE_SIZE}; wr[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0}; }
  vkUpdateDescriptorSets(dev,3,wr,0,0);
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType=VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount=2; VkQueryPool qp; CK(vkCreateQueryPool(dev,&qpi,0,&qp));
  auto run=[&](int mode)->double{
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; a.commandPool=cp; a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount=1; VkCommandBuffer c; CK(vkAllocateCommandBuffers(dev,&a,&c));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(c,&bi);
    vkCmdFillBuffer(c,bufs[0],0,VK_WHOLE_SIZE,0x3f800000); vkCmdFillBuffer(c,bufs[1],0,VK_WHOLE_SIZE,0x3f800000); vkCmdFillBuffer(c,bufs[2],0,VK_WHOLE_SIZE,0);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);
    vkCmdResetQueryPool(c,qp,0,2); vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipes[mode]); vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qp,0);
    if(mode==0){ for(uint32_t i=0;i<nph;i++){ uint32_t pc[4]={total,i,nph,0}; vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc); vkCmdDispatch(c,nwg,1,1);
        static int bm=getenv("BARMODE")?atoi(getenv("BARMODE")):0;
        VkAccessFlags af=bm==1?(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT):bm==2?(VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT):VK_ACCESS_SHADER_WRITE_BIT;
        VkAccessFlags af2=bm==0?(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT):af;
        VkPipelineStageFlags st=bm==1?(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT):bm==2?VK_PIPELINE_STAGE_ALL_COMMANDS_BIT:VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkMemoryBarrier m2{VK_STRUCTURE_TYPE_MEMORY_BARRIER,0,af,af2};
        vkCmdPipelineBarrier(c,st,st,0,1,&m2,0,0,0,0);} }
    else { uint32_t pc[4]={total,0,nph,0}; vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc); vkCmdDispatch(c,nwg,1,1); }
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,qp,1); vkEndCommandBuffer(c);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&c; CK(vkQueueSubmit(q,1,&si,0)); CK(vkQueueWaitIdle(q));
    uint64_t t[2]; CK(vkGetQueryPoolResults(dev,qp,0,2,16,t,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)); vkFreeCommandBuffers(dev,cp,1,&c);
    return (double)(t[1]-t[0])*tsp*1e-3/nph; };
  for(int mode=0;mode<2;mode++){ run(mode); std::vector<double> v; for(int r=0;r<7;r++) v.push_back(run(mode)); std::sort(v.begin(),v.end());
    printf("%s nwg=%u phases=%u: min %.2f us/phase  med %.2f us/phase\n",mode?"persistent":"chain     ",nwg,nph,v[0],v[3]); }
  return 0; }
