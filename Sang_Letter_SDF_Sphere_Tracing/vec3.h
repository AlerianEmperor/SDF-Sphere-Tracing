#ifndef _VEC3_H_
#define _VEC3_H_
#include <iostream>

#define min(x, y) x < y ? x : y
#define max(x, y) x > y ? x : y
#define abs(x) ((x)<0 ? -(x) : (x))
#define square(x) x * x
#define eps 1e-4
#define inf 1e10
#define pi 3.1415926
#define ipi 0.3183098
#define i2pi 0.1591549


using namespace std;

double inline __declspec (naked) __fastcall sqrt14(double n)
{
	_asm fld qword ptr[esp + 4]
		_asm fsqrt
	_asm ret 8

}



template <typename T>
T clamp(const T& n, const T& lower, const T& upper)
{
	return max(lower, min(n, upper));
}

struct vec3
{
	vec3() : x(0), y(0), z(0) {}
	vec3(float v) : x(v), y(v), z(v) {}
	vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	float x;
	float y;
	float z;

	float maxc() { return max(x, max(y, z)); }
	float minc() { return min(x, min(y, z)); }
	float operator[](const int& i) const { return (&x)[i]; }
	friend vec3 operator+(const vec3& a, const vec3& b) { return{ a.x + b.x, a.y + b.y, a.z + b.z }; }
	friend vec3 operator-(const vec3& a, const vec3& b) { return{ a.x - b.x, a.y - b.y, a.z - b.z }; }
	friend vec3 operator*(const vec3& a, const vec3& b) { return{ a.x * b.x, a.y * b.y, a.z * b.z }; }
	friend vec3 operator*=(const vec3& a, const float& v) { return{ a.x * v, a.y * v, a.z * v }; }
	friend vec3 operator/(const vec3& a, const vec3& b) { return{ a.x / b.x, a.y / b.y, a.z / b.z }; }
	friend vec3 operator/=(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }
	friend vec3 operator-(const vec3& a) { return{ -a.x, -a.y, -a.z }; }

	friend vec3 operator/(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }
	friend vec3 operator/(const vec3& a, const int& v) { return{ a.x / v, a.y / v, a.z / v }; }
	friend vec3 operator*(const vec3& a, const float& v) { return{ a.x * v, a.y * v, a.z * v }; }
	friend vec3 operator*(const float& v, const vec3& a) { return{ a.x * v, a.y * v, a.z * v }; }
	vec3 __fastcall  operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
	vec3 __fastcall  operator*=(const float& value) { x *= value; y *= value; z *= value; return *this; }

	friend std::ostream& operator<<(std::ostream&os, const vec3& v)
	{
		os << v.x << "  " << v.y << "  " << v.z << "  " << "\n";
		return os;
	}

	vec3 __fastcall norm() const
	{
		const float l = 1.0 / sqrt14(x*x + y*y + z*z); return *this * l;
	}
	float __fastcall dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
	float length() const { return sqrt14(x * x + y * y + z * z); }
	float length2() const { return x * x + y * y + z * z; }
	vec3 operator%(const vec3& v) const { return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
	vec3 __fastcall  cross(const vec3& v) const { return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
};


vec3 clampvec(vec3& v, const float& start, const float& end)
{
	v.x = clamp(v.x, start, end);
	v.y = clamp(v.y, start, end);
	v.z = clamp(v.z, start, end);

	return v;
}


#endif // !_VEC3_H_