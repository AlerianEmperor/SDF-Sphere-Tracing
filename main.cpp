#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>

using namespace std;

#define pi 3.1415926

double inline __declspec (naked) __fastcall sqrt14(double n)
{
	_asm fld qword ptr[esp + 4]
		_asm fsqrt
	_asm ret 8
}

thread_local uint32_t s_RndState = 1;
static const double imax = 1.0f / UINT32_MAX;
static const double irand_max = 1.0f / RAND_MAX;
double randf()
{
	uint32_t x = s_RndState;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 15;
	s_RndState = x;
	return x * imax;
}

double max(double& a, double& b) { return a > b ? a : b; }
struct vec3
{
	vec3() {}
	vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
	double x = 0.0f, y = 0.0f, z = 0.0f;
	friend vec3 operator+(const vec3& a, const vec3& b) { return{ a.x + b.x, a.y + b.y, a.z + b.z }; }
	friend vec3 operator-(const vec3& a, const vec3& b) { return{ a.x - b.x, a.y - b.y, a.z - b.z }; }
	friend vec3 operator*(const vec3& a, const vec3& b) { return{ a.x * b.x, a.y * b.y, a.z * b.z }; }
	friend vec3 operator/(const vec3& a, const vec3& b) { return{ a.x / b.x, a.y / b.y, a.z / b.z }; }
	friend vec3 operator-(const vec3& a) { return{ -a.x, -a.y, -a.z }; }
	//friend vec3 operator/(const vec3& a, const double& v) { return{ a.x / v, a.y / v, a.z / v }; }	
	friend vec3 operator*(const vec3& a, const double& v) { return{ a.x * v, a.y * v, a.z * v }; }
	friend vec3 operator*(const double& v, const vec3& a) { return{ a.x * v, a.y * v, a.z * v }; }
	vec3 __fastcall norm() const { const double l = 1.0 / sqrt14(x*x + y*y + z*z); return *this * l; }
	double __fastcall dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
	double length() const { return sqrt14(x * x + y * y + z * z); }
	vec3 __fastcall  cross(const vec3& v) const { return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
};

struct Ray
{
	vec3 o, d;
	Ray() {}
	Ray(vec3 o_, vec3 d_) :o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR, LITE };

struct Sphere
{
	double rad;       // radius 
	vec3 p, e, c;      // position, emission, color 
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive) 
	Sphere(double rad_, vec3 p_, vec3 e_, vec3 c_, Refl_t refl_) :rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	double intersect(const Ray &r) const
	{
		// returns distance, 0 if nohit 
		vec3 op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
		double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0)
			return 0;
		else 
			det = sqrt14(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

//midx(50)
//midy(40)
//z(190)

Sphere spheres[] = {//Scene: radius, position, emission, color, material 
	Sphere(1e5, vec3(1e5 + 1,40.8,81.6), vec3(),vec3(.75f,.25f,.25f),DIFF),//Left 
	Sphere(1e5, vec3(-1e5 + 99,40.8,81.6),vec3(),vec3(.25f,.25f,.75f),DIFF),//Rght 
	Sphere(1e5, vec3(50,40.8, 1e5),     vec3(),vec3(.75f,.75f,.75f),DIFF),//Back 
	//Sphere(1e5, vec3(50,40.8,-1e5 + 170), vec3(),vec3(),           DIFF),//Frnt 
	Sphere(1e5, vec3(50, 1e5, 81.6),    vec3(),vec3(.75f,.75f,.75f),DIFF),//Botm 
	Sphere(1e5, vec3(50,-1e5 + 81.6,81.6),vec3(),vec3(.75f,.75f,.75f),DIFF),//Top 
	Sphere(16.5,vec3(27,16.5,47),       vec3(),vec3(1,1,1)*.999, SPEC),//Mirr 
	Sphere(16.5,vec3(73,16.5,78),       vec3(),vec3(1,1,1)*.999, SPEC),//Glas 
	Sphere(600, vec3(50,681.6 - .27,81.6),vec3(12,12,12),  vec3(), LITE) //Lite 
};

double max_reflectance(vec3& v) { double a = max(v.x, v.y); return max(a, v.z); }

double clamp(double x) { return x<0 ? 0 : x>1 ? 1 : x; }
int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
bool intersect(const Ray &r, double &t, int &id) {
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
	return t < inf;
}

void onb(vec3& n, vec3& u, vec3& v)
{
	if (n.z < -0.9999999f) // Handle the singularity
	{
		u = vec3(0.0f, -1.0f, 0.0f);
		v = vec3(-1.0f, 0.0f, 0.0f);
		return;
	}
	else
	{
		const double a = 1.0f / (1.0f + n.z);
		const double b = -n.x * n.y * a;
		u = vec3(1.0f - n.x * n.x * a, b, -n.x);
		v = vec3(b, 1.0f - n.y * n.y * a, -n.y);
	}
}

vec3 Radiance(const Ray& r)
{
	Ray new_ray = r;

	vec3 L(0.0f, 0.0f, 0.0f);
	vec3 T(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 60; ++i)
	{
		double t;
		int id = 0;
		if (!intersect(new_ray, t, id))
			return vec3();
		//cout << id << "\n";
		//if (spheres[id].c.x != 0)
		//	cout << spheres[id].c.x << "\n";
		//return spheres[id].c;

	
		vec3 x = new_ray.o + new_ray.d * t;
		vec3 n = (x - spheres[id].p).norm();
	
		

		//vec3 nl = n.dot(r.d) < 0.0f ? n : -n;

		bool into = n.dot(r.d) < 0.0f;

		vec3 nl = into ? n : -n;

		vec3 f = spheres[id].c;


		double p = max_reflectance(f);

		if (i >= 7)
		{
			if (randf() < p)
				T = T * (1.0f / p);
			else
				return vec3(0.0f, 0.0f, 0.0f);//spheres[id].e;
		}
		if (spheres[id].refl == LITE)
		{
			//cout << "light\n";
			L = L + T * 2;// spheres[i].e;

			return L;
		}
		else if (spheres[id].refl == DIFF)
		{
			T = T * f;
			double r1 = randf(), r2 = randf();
			double r2s = sqrt14(r1);
			double theta = 2.0f * pi * r2;

			double c = cosf(theta), s = sinf(theta);

			vec3 u, v;

			onb(nl, u, v);

			//vec3 direction(u * r2s * c + v * r2s * s + nl * sqrt14(1.0f - r1));

			vec3 direction(r2s * (u * c + v * s) + nl * sqrt14(1.0f - r1));

			new_ray = Ray(x, direction.norm());

			//T = T * f;
		}
		else if (spheres[id].refl == SPEC)
		{
			T = T * f;
			new_ray = Ray(x, (r.d - 2.0f * n * n.dot(r.d)).norm());
		}
		/*else if (spheres[id].refl == REFR)
		{
			double nc = 1.0f;
			double nt = 1.5f;


		}*/

	}
}

void main()
{
	int width = 480;
	int height = 270;

	double iWidth = 1.0f / width;
	double iHeight = 1.0f / height;

	double aspect_ratio = (double)(width) / height;
	//cout << iWidth << "\n";
	//cout << aspect_ratio << "\n";

	int ns = 16;
	double ins = 1.0f / ns;
	
	const double tan_theta = tanf(60.0f);

	Ray cam(vec3(50, 42, 245.6), vec3(0, 0, -1));

	//cam.o = cam.o;// +cam.d * 140.0f;

	const vec3 w = -cam.d;
	const vec3 up = vec3(0, 1, 0);
	const vec3 u = up.cross(w).norm();
	const vec3 v = w.cross(u);

	vector<vec3> color;

	color.resize(width * height);

	for (int j = 0; j < height; ++j)
	{
		#pragma omp parallel for schedule(guided)
		for (int i = 0; i < width; ++i)
		{
			fprintf(stderr, "\rRendering (%d spp) %5.2f%%", ns, 100.0f * j / (height - 1));

			vec3 L(0.0f, 0.0f, 0.0f);
			for (int s = 0; s < ns; ++s)
			{
				double p = ((double)i + randf()) * iWidth;
				double q = ((double)j + randf()) * iHeight;

				p = (2.0f * p - 1.0f) * aspect_ratio * tan_theta;
				q = (1.0f - 2.0f * q) * tan_theta;

				//Ray r = Ray(cam.o + u * p + v * q, cam.d);
				
				Ray r = Ray(cam.o, (u * p + v * q - w).norm());

				vec3 x = Radiance(r);

				L = L + x;//vec3(clamp(x.x), clamp(x.y), clamp(x.z));
			}
			//L = L * ins;

			color[j * width + i] = L * ins;
		}
	}

	ofstream ofs("Result.ppm");

	ofs << "P3\n" << width << " " << height << "\n255\n";

	for (int i = 0; i < width * height; ++i)
	{
		vec3 c = color[i];
	
		c = vec3(toInt(c.x), toInt(c.y), toInt(c.z));//255.99f;
		//cout << c.x << "\n";
		ofs << c.x << " " << c.y << " " << c.z << "\n";
	}
	

	//getchar();
}

/*struct onb
{
vec3 u, v, w;
onb(const vec3& n) : w(n)
{
if (n.z < -0.9999999f) // Handle the singularity
{
u = vec3(0.0f, -1.0f, 0.0f);
v = vec3(-1.0f, 0.0f, 0.0f);
return;
}
else
{
const double a = 1.0f / (1.0f + n.z);
const double b = -n.x * n.y * a;
u = vec3(1.0f - n.x * n.x * a, b, -n.x);
v = vec3(b, 1.0f - n.y * n.y * a, -n.y);
}
}
};*/

