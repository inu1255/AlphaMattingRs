#version 450

layout(location=0)out vec4 f_color;

layout(set=0,binding=0)uniform sampler2D triState;
layout(push_constant)uniform PushConstants{
  vec2 scale;
}push_constants;

void main(){
  f_color=texture(triState,(gl_FragCoord.xy)/push_constants.scale);
}
