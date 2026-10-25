#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Packed logical indices <-> strided device allocation. Preserve padding.
struct TensorCopy {
 std::vector<size_t> shape,strides;size_t elem,count=1,run=1;
 TensorCopy(std::vector<size_t> dims,std::vector<size_t> steps,size_t bytes):shape(std::move(dims)),strides(std::move(steps)),elem(bytes){
  if(shape.size()!=strides.size()||elem==0)throw std::runtime_error("Invalid copy layout");
  for(auto d:shape){if(!d)throw std::runtime_error("Empty copy dimension");count*=d;}
  for(int d=int(shape.size())-1;d>=0;--d){
   if(shape[d]==1)continue;
   if(strides[d]!=run)break;
   run*=shape[d];
  }
 }
 size_t offset(size_t linear)const{size_t off=0;for(int d=int(shape.size())-1;d>=0;--d){off+=(linear%shape[d])*strides[d];linear/=shape[d];}return off*elem;}
 void copy(void* device,void* packed,size_t start,size_t length,bool write,bool fast)const{
  if(start>count||length>count-start)throw std::runtime_error("Copy range outside tensor");
  auto* d=static_cast<uint8_t*>(device);auto* p=static_cast<uint8_t*>(packed);
  size_t block=fast?run:1;
  for(size_t done=0;done<length;){size_t at=start+done,n=std::min(length-done,block-at%block);
   auto* remote=d+offset(at);auto* local=p+done*elem;
   if(write)std::memcpy(remote,local,n*elem);else std::memcpy(local,remote,n*elem);
   done+=n;
  }
 }
};
