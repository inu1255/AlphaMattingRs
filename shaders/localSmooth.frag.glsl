#version 450

layout(location=0)out vec4 f_color;
layout(set=0,binding=0)uniform sampler2D oriState;
layout(set=0,binding=1)uniform sampler2D triState;
layout(set=0,binding=2)uniform sampler2D foreState;
layout(set=0,binding=3)uniform sampler2D backState;
layout(set=0,binding=4)uniform sampler2D acState;
layout(push_constant)uniform PushConstants{
  vec2 scale;
}push_constants;

vec3 getOri() {
  vec4 color = texture(oriState, (gl_FragCoord.xy) / push_constants.scale);
  return color.xyz;
}

vec3 getOri(vec2 coord) {
  vec4 color = texture(oriState, coord);
  return color.xyz;
}

vec3 getTri() {
  vec4 color = texture(triState, (gl_FragCoord.xy) / push_constants.scale);
  return color.xyz;
}

vec3 getTri(vec2 coord) {
  return texture(triState, coord).xyz;
}

float getAlpha(vec2 coord) {
  return texture(acState, coord).x;
}

float getConfidence(vec2 coord) {
  return texture(acState, coord).y;
}

vec3 getFore(vec2 coord) {
  return texture(foreState, coord).xyz;
}

vec3 getBack(vec2 coord) {
  return texture(backState, coord).xyz;
}

float estimAlpha(vec3 color, vec3 fColor, vec3 bColor) {
  float d = distance(fColor, bColor);
  return dot((color - bColor), (fColor - bColor)) / (d * d);
}

float m_p(vec2 coord, vec3 fColor, vec3 bColor) {
  vec3 color = getOri(coord);
  float alpha = estimAlpha(color, fColor, bColor);
  return length(color - alpha * fColor - (1.0 - alpha) * bColor);
}

float smooth1() {
  float sigma2 = 100.0 / (9.0 * 3.1415926);
  vec3 accumWcUpF = vec3(0.0, 0.0, 0.0), accumWcUpB = vec3(0.0, 0.0, 0.0);
  float accumWcDownF = 0.0, accumWcDownB = 0.0;
  float accumWfbUp = 0.0, accumWfbDown = 0.0;
  float confidence00 = getConfidence(gl_FragCoord.xy / push_constants.scale);
  float alpha00 = getAlpha(gl_FragCoord.xy / push_constants.scale);
  float accumWaUp = 0.0, accumWaDown = 0.0;
  for (int i = -10; i < 11; i ++) {
    for (int j = -10; j < 11; j ++) {
      vec2 offset = vec2(float(i), float(j));
      vec2 coord = (gl_FragCoord.xy + offset) / push_constants.scale;
      float d = distance(offset, vec2(0.0, 0.0));
      if (coord.x > 0.0 && coord.x < 1.0
          && coord.y > 0.0 && coord.y < 1.0
          && d <= 3.0 * sigma2) {
        // aqr
        float alpha = getAlpha(coord),
        // fqr
          confidence = getConfidence(coord);
        float g = exp(- d*d / 2.0 / sigma2);
        float wc = 0.0;
        if (i == 0 && j == 0) {
          wc = g * confidence;
        }
        else {
          wc = g * confidence * abs(alpha - alpha00);
        }
        vec3 foreColor = getFore(coord),
          backColor = getBack(coord);
        float wca = wc * alpha;
        accumWcUpF += wca * foreColor;
        accumWcUpB += (wc - wca) * backColor;
        accumWcDownF += wc * alpha;
        accumWcDownB += wc - wca;

        float wfbq = confidence * alpha * (1.0 - alpha);
        accumWfbUp += wfbq * distance(foreColor, backColor);
        accumWfbDown += wfbq;

        float theta = 0.0;
        if (getTri(coord).x == 0.0 || getTri(coord).x == 1.0) {
          theta = 1.0;
        }
        float wa = confidence * g + theta;
        accumWaUp += wa * alpha;
        accumWaDown += wa;
      }
    }
  }
  vec3 fp = accumWcUpF / (accumWcDownF + 1e-10),
    bp = accumWcUpB / (accumWcDownB + 1e-10);
  float dfb = accumWfbUp / (accumWfbDown + 1e-10);
  float confidenceP = min(1.0, distance(fp, bp) / dfb) * exp(-10.0 * m_p(gl_FragCoord.xy/push_constants.scale, fp, bp));
  float alphaP = accumWaUp / (accumWaDown + 1e-10);
  alphaP = max(0.0, min(1.0, alphaP));
  float alphaFinal = confidenceP * estimAlpha(getOri(), fp, bp) + (1.0 - confidenceP) * alphaP;

  return alphaFinal;
}

void getAlpha() {
  // vec4 color = texture(oriState, (gl_FragCoord.xy) / push_constants.scale);
  // if (color.z == 0.0)
  //   f_color = color;
  // else {
  //   float x = getTri().x;
  //   if (x == 1.0) {
  //     f_color = vec4(color.xyz, 1.0);
  //   }
  //   else if(x == 0.0) {
  //     f_color = vec4(color.xyz, 0.0);
  //   }
  //   else {
  //     x = smooth1();
  //     // f_color = vec4(x, x, x, 1.0);
  //     if (x > 0.3) {
  //       f_color = vec4(color.xyz, 1.0);
  //     }
  //     else {
  //       f_color = vec4(color.xyz, 0.0);
  //     }
  //   }
  // }
  vec3 color = getTri();
  if (color.x == 0.0 || color.x == 1.0) {
    f_color = vec4(color, 1.0);
  }
  else {
    float alpha = smooth1();
    f_color = vec4(alpha, alpha, alpha, 1.0);
  }
}

void main() {
  getAlpha();
}
