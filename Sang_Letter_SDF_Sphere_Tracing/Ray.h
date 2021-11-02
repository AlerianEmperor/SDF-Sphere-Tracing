#ifndef _RAY_H_
#define _RAY_H_
#include "vec3.h"

struct Ray
{
	Ray() {}
	Ray(vec3 o_, vec3 d_) : o(o_), d(d_) {}
	vec3 o;
	vec3 d;
};

#endif // !_RAY_H_

