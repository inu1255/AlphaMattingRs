#version 450

layout(location=0)out vec4 f_color;

layout(set=0,binding=0)uniform sampler2D oriState;
layout(set=0,binding=1)uniform sampler2D triState;
layout(push_constant)uniform PushConstants{
  vec2 scale;
  float kC;
  float kI;
}push_constants;

vec3 getOri(){
  vec4 color=texture(oriState,(gl_FragCoord.xy)/push_constants.scale);
  return color.xyz;
}

vec3 getOri(vec2 coord){
  vec4 color=texture(oriState,coord);
  return color.xyz;
}

vec3 getTri(){
  vec4 color=texture(triState,(gl_FragCoord.xy)/push_constants.scale);
  return color.xyz;
}

vec4 getTri(vec2 coord){
  return texture(triState,coord);
}

void main(){
  float tri=getTri().x;
  f_color=texture(triState,(gl_FragCoord.xy)/push_constants.scale);
  
  // unknown region
  if(tri>0.&&tri<1.){
    float x=gl_FragCoord.x,
    y=gl_FragCoord.y;
    
    const float kI=push_constants.kI;
    
    vec3 curColor=getOri();
    for(float i=-kI;i<kI;i+=1.){
      for(float j=-kI;j<kI;j+=1.){
        float w=gl_FragCoord.x+i;
        float h=gl_FragCoord.y+j;
        if(w>0.&&w<push_constants.scale.x
        &&h>0.&&h<push_constants.scale.y){
          vec2 coord=vec2(w,h);
          vec3 color=getOri(coord/push_constants.scale);
          vec4 triColor=getTri(coord/push_constants.scale);
          if((triColor.x==0.||triColor.x==1.)
          &&distance(coord,gl_FragCoord.xy)<kI
          &&distance(color,curColor)<push_constants.kC){
            f_color=triColor;
          }
        }
      }
    }
  }
}
