#pragma once
#include <arpa/inet.h>
#include <cerrno>
#include <cstddef>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>

// One authenticated connection per worker; no reconnect/replay of mutations.
struct SocketRpc {
 int listener=-1,client=-1;
 ~SocketRpc(){if(client>=0)close(client);if(listener>=0)close(listener);}
 void listen_on(const std::string& name){
  sockaddr_un addr{};addr.sun_family=AF_UNIX;
  require(!name.empty()&&name.size()<sizeof(addr.sun_path)-1,"Invalid RPC socket name");
  std::memcpy(addr.sun_path+1,name.data(),name.size());
  listener=socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0);require(listener>=0,"RPC socket failed");
  require(bind(listener,(sockaddr*)&addr,offsetof(sockaddr_un,sun_path)+1+name.size())==0,"RPC bind failed");
  require(listen(listener,1)==0,"RPC listen failed");
 }
 void accept_one(){
  pollfd p{listener,POLLIN,0};int rc;do{rc=poll(&p,1,180000);}while(rc<0&&errno==EINTR);
  require(rc>0,"RPC connection timeout");client=accept4(listener,nullptr,nullptr,SOCK_CLOEXEC);require(client>=0,"RPC accept failed");
  timeval timeout{900,0};require(setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout))==0,"RPC timeout failed");
  require(setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout))==0,"RPC timeout failed");
 }
 void transfer(void* buffer,size_t length,bool sending){
  auto* p=static_cast<uint8_t*>(buffer);
  while(length){ssize_t n=sending?send(client,p,length,MSG_NOSIGNAL):recv(client,p,length,0);
   if(n<0&&errno==EINTR)continue;require(n>0,"RPC disconnected or timed out; no replay");p+=n;length-=n;}
 }
 J receive(){
  uint32_t n;transfer(&n,4,false);n=ntohl(n);require(n>0&&n<=(2u<<20),"Invalid RPC command size");
  Bytes b(n);transfer(b.data(),b.size(),false);return J::parse(b);
 }
 void reply(const J& response,Bytes& payload){
  auto text=response.dump();uint32_t n=htonl(text.size());transfer(&n,4,true);
  transfer(text.data(),text.size(),true);transfer(payload.data(),payload.size(),true);
 }
};
