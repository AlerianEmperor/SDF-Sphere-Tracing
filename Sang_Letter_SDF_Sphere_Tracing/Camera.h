#ifndef _CAMERA_H_
#define _CAMERA_H_
#include "Ray.h"

struct Camera
{
	vec3 look_from;
	vec3 look_at;
	int width;
	int height;
	float aspect_ratio;
	vec3 u;
	vec3 v;
	vec3 w;
	vec3 up;
	float tan_theta;

	Camera() {}
	Camera(vec3 look_from_, vec3 look_at_,  int width_, int height_) : look_from(look_from_), look_at(look_at_), width(width_), height(height_) 
	{
		w = (look_from - look_at).norm();
		up = vec3(0, 1, 0);
		u = up.cross(w).norm();
		v = w.cross(u);
		//tan_theta = tanf(32.0f * pi / 180.0f);//40
	
		tan_theta = tanf(32.0f * pi / 180.0f);

		aspect_ratio = float(width) / float(height);
	}
	
	Ray __fastcall generate_ray(float& p, float& q)
	{
		p = (2.0f * p - 1.0f) * aspect_ratio * tan_theta;
		q = (1.0f - 2.0f * q) * tan_theta;

		return Ray(look_from, u * p + v * q - w);
	}
};


#endif // !_CAMERA_H_

