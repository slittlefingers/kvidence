#include "tensor_copy.h"
#include <iostream>
#include <numeric>
#include <random>

static size_t scalar_offset(size_t at,const std::vector<size_t>& shape,const std::vector<size_t>& strides){
 size_t result=0;for(int d=int(shape.size())-1;d>=0;--d){result+=(at%shape[d])*strides[d];at/=shape[d];}return result;
}
int main(){
 std::mt19937 rng(20261019);size_t cases=0;
 for(size_t elem:{1,4})for(int trial=0;trial<1000;trial++){
  int rank=1+rng()%5;std::vector<size_t> shape(rank),strides(rank),order(rank);std::iota(order.begin(),order.end(),0);
  for(auto& d:shape)d=1+rng()%5;
  if(trial%3==0)std::shuffle(order.begin(),order.end(),rng);
  size_t stride=1;
  for(int k=rank-1;k>=0;--k){auto d=order[k];strides[d]=stride;stride*=shape[d];if(trial%3==1)stride+=rng()%4;}
  for(int d=0;d<rank;d++)if(shape[d]==1&&trial%7==0)strides[d]=0;
  TensorCopy copy(shape,strides,elem);size_t last=0;for(int d=0;d<rank;d++)last+=(shape[d]-1)*strides[d];
  size_t alloc=(last+1)*elem+32;std::vector<uint8_t> raw(alloc);for(auto& b:raw)b=rng()%256;
  std::vector<uint8_t> slow(copy.count*elem),fast(slow.size());
  for(size_t i=0;i<copy.count;i++)std::memcpy(slow.data()+i*elem,raw.data()+scalar_offset(i,shape,strides)*elem,elem);
  copy.copy(raw.data(),fast.data(),0,copy.count,false,true);if(fast!=slow)throw std::runtime_error("read differs");
  size_t start=rng()%(copy.count+1),length=rng()%(copy.count-start+1);
  std::vector<uint8_t> input(length*elem);for(auto& b:input)b=rng()%256;
  auto expected=raw,actual=raw;
  for(size_t i=0;i<length;i++)std::memcpy(expected.data()+scalar_offset(start+i,shape,strides)*elem,input.data()+i*elem,elem);
  copy.copy(actual.data(),input.data(),start,length,true,true);if(actual!=expected)throw std::runtime_error("partial write/padding differs");
  auto scalar=raw;copy.copy(scalar.data(),input.data(),start,length,true,false);if(scalar!=expected)throw std::runtime_error("reference differs");
  cases++;
 }
 TensorCopy whole({2,1,4},{4,99,1},4);if(whole.run!=8)throw std::runtime_error("contiguous singleton layout");
 std::cout<<"PASS "<<cases<<" strided/contiguous copy cases\n";
}
