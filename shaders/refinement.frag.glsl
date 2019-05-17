#version 450

layout(location=0)out vec4 refine_f;
layout(location=1)out vec4 refine_b;
layout(location=2)out vec4 refine_ac;
layout(set=0,binding=0)uniform sampler2D oriState;
layout(set=0,binding=1)uniform sampler2D triState;
layout(set=0,binding=2)uniform sampler2D fbState;
layout(push_constant)uniform PushConstants{
  vec2 scale;
  int isTest;
}push_constants;

struct point {
  float x;
  float y;
  float mp;
};

struct rTuple {
  vec3 fColor;
  vec3 bColor;
  float alpha;
  float confidence;
};

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

vec2 getFCoord() {
  return texture(fbState, (gl_FragCoord.xy) / push_constants.scale).xy;
}

vec2 getBCoord() {
  return texture(fbState, (gl_FragCoord.xy) / push_constants.scale).zw;
}

vec2 getFCoord(vec2 coord) {
  return texture(fbState, coord).xy;
}

vec2 getBCoord(vec2 coord) {
  return texture(fbState, coord).zw;
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

float m_p(vec3 color, vec3 fColor, vec3 bColor) {
  float alpha = estimAlpha(color, fColor, bColor);
  return length(color - alpha * fColor - (1.0 - alpha) * bColor);
}

float sigma2(vec2 coord) {
  float res = 0.0;
  int neiCount = 0;
  vec3 color = getOri(coord);
  for (int i = -2; i < 3; i ++) {
    for (int j = -2; j < 3; j ++) {
      float xn = coord.x + (float(i) / push_constants.scale.x),
        yn = coord.y + (float(j) / push_constants.scale.y);
      if (xn > 0.0 && xn < 1.0
        && yn > 0.0 && yn < 1.0) {
          vec3 colorn = getOri(vec2(xn, yn));
          float d = distance(colorn, color);
          res += d * d;
          neiCount ++;
        }
    }
  }
  return res / (float(neiCount) + 1e-10);
}

rTuple refine() {
  int count = 0;
  point minMp0, minMp1, minMp2;
  minMp0.mp = 1e10; minMp1.mp = 1e10; minMp2.mp = 1e10;
  for (int i = -5; i < 6; i ++) {
    for (int j = -5; j < 6; j ++) {
      float x = gl_FragCoord.x + float(i),
        y = gl_FragCoord.y + float(j);
      vec2 coord = vec2(x, y) / push_constants.scale;

      if (x > 0.0 && x < push_constants.scale.x
        && y > 0.0 && y < push_constants.scale.y
        && getTri(coord).x > 0.0 &&  getTri(coord).x < 1.0) {

          vec3 fColor = getOri(getFCoord(coord));
          vec3 bColor = getOri(getBCoord(coord));
          float mp = m_p(coord, fColor, bColor);
          if (mp < minMp0.mp) {
            minMp2 = minMp1;
            minMp1 = minMp0;
            // minMp2.x = minMp1.x;
            // minMp2.y = minMp1.y;
            // minMp2.mp = minMp1.mp;
            // minMp1.x = minMp0.x;
            // minMp1.y = minMp0.y;
            // minMp1.mp = minMp0.mp;

            minMp0.x = coord.x;
            minMp0.y = coord.y;
            minMp0.mp = mp;
            count ++;
          }
          else if (mp < minMp1.mp) {
            minMp2 = minMp1;
            // minMp2.x = minMp1.x;
            // minMp2.y = minMp1.y;
            // minMp2.mp = minMp1.mp;

            minMp1.x = coord.x;
            minMp1.y = coord.y;
            minMp1.mp = mp;
            count ++;
          }
          else if (mp < minMp2.mp) {
            minMp2.x = coord.x;
            minMp2.y = coord.y;
            minMp2.mp = mp;
            count ++;
          }
        }
    }
  }

  // count must > 0 because self is included
  count = count > 3 ? 3 : count;

  vec2 coord = vec2(minMp0.x, minMp0.y);
  vec2 fCoord = getFCoord(coord),
    bCoord = getBCoord(coord);
  vec3 fg = getOri(fCoord),
    bg = getOri(bCoord);
  float sf = sigma2(fCoord),
    sb = sigma2(bCoord);

  if (count > 1) {
    coord = vec2(minMp1.x, minMp1.y);
    fCoord = getFCoord(coord);
    bCoord = getBCoord(coord);
    fg += getOri(fCoord);
    bg += getOri(bCoord);
    sf += sigma2(fCoord);
    sb += sigma2(bCoord);
  }
  if (count > 2) {
    coord = vec2(minMp2.x, minMp2.y);
    fCoord = getFCoord(coord);
    bCoord = getBCoord(coord);
    fg += getOri(fCoord);
    bg += getOri(bCoord);
    sf += sigma2(fCoord);
    sb += sigma2(bCoord);
  }
  fg = fg / float(count);
  bg = bg / float(count);
  sf = sf / float(count);
  sb = sb / float(count);

  vec3 color = getOri();
  rTuple r;
  float d = distance(color, fg);
  float df = d * d;
  d = distance(color, bg);
  float db = d * d;

  if (df < sf) { r.fColor = color; }
  else { r.fColor = fg; }
  if (db < sb) { r.bColor = color; }
  else { r.bColor = bg; }

  if (r.fColor != r.bColor) { r.confidence = exp(-10.0 * m_p(color, fg, bg)); }
  else { r.confidence = 1e-8; }
  r.alpha = max(0.0, min(1.0, estimAlpha(color, r.fColor, r.bColor)));
  // r.alpha = estimAlpha(color, fg, bg);
  // r.alpha = estimAlpha(getOri(), getOri(getFCoord()), getOri(getBCoord()));

  return r;
}

void writeFB() {
  float triColor = getTri().x;
  if (triColor == 0.0 || triColor == 1.0) {
    // fore color texture
    refine_f = vec4(getOri(), 1.0);
    // back color texture
    refine_b = vec4(getOri(), 1.0);
    // alpha and confidence
    refine_ac = vec4(triColor, 1.0, 1.0, 1.0);
  }
  else {
    rTuple r = refine();
    // fore color texture
    refine_f = vec4(r.fColor, 1.0);
    // back color texture
    refine_b = vec4(r.bColor, 1.0);
    // alpha and confidence
    refine_ac = vec4(r.alpha, r.confidence, 1.0, 1.0);
  }
}

void testAlpha() {
  float triColor = getTri().x;
  if (triColor == 0.0 || triColor == 1.0) {
    refine_f = vec4(getTri(), 1.0);
  }
  else {
    rTuple r = refine();
    // refine_f = vec4(getOri(getBCoord()), 1.0);
    // refine_f = vec4(r.fColor, 1.0);
    // float s = sigma2(getBCoord());
    // refine_f = vec4(s, s, s, 1.0);
    refine_f = vec4(r.alpha, r.alpha, r.alpha, 1.0);
  }
}

void main() {
  if (push_constants.isTest == 0) {
    writeFB();
  }
  else {
    testAlpha();
  }
}
