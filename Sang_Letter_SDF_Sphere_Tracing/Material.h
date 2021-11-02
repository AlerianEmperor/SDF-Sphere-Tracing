#ifndef _MATERIAL_H_
#define _MATERIAL_H_
#include "vec3.h"
#include "Rnd.h"
#include "fasttrigo.h"
#define tau 6.28318530
//#define max(x, y) x > y ? x : y
//#define min(x, y) x < y ? x : y

class Material
{
public:
	virtual vec3 bsdf_sample(const vec3& dir_in, const vec3& n) = 0;
	virtual bool isSpecular() = 0;
	//vec3 color;
};

class Diffuse : public Material
{
public:
	//Diffuse() {}
	Diffuse(vec3 c) : color(c) {}
	vec3 color;
	vec3 bsdf_sample(const vec3& dir_in, const vec3& n) 
	{
		float u1 = randf();
		
		float r = sqrt14(u1);

		float theta = tau * randf();

		float c, s;
		FTA::sincos(theta, &c, &s);

		float x = r * c;
		float y = r * s;
		//float x = r * cosf(theta);
		//float y = r * sinf(theta);

		return vec3(x, y, sqrt14(max(0.0f, 1.0f - u1)));
	}
	bool isSpecular()
	{
		return false;
	}
};

class Mirror : public Material
{
public:
	Mirror() {}
	vec3 bsdf_sample(const vec3& dir_in, const vec3& n)
	{
		return dir_in - 2.0f * dir_in.dot(n);
	}
	bool isSpecular()
	{
		return true;
	}
};

#endif // !_MATERIAL_H_

