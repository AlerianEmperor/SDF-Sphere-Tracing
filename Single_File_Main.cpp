#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <omp.h> 
#include <stdlib.h>
#include <string>
#include <time.h>
#include "fasttrigo.h"

#define min(x, y) x < y ? x : y
#define max(x, y) x > y ? x : y
#define abs(x) ((x)<0 ? -(x) : (x))
#define square(x) x * x
#define eps 1e-4
#define inf 1e10
#define pi 3.1415926
#define ipi 0.3183098
#define i2pi 0.1591549
#define tau 6.28318530
//#define max(x, y) x > y ? x : y
//#define min(x, y) x < y ? x : y

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

struct Ray
{
	Ray() {}
	Ray(vec3 o_, vec3 d_) : o(o_), d(d_) {}
	vec3 o;
	vec3 d;
};

using namespace std;

thread_local uint32_t s_RndState = 1;
static const float imax = 1.0f / UINT32_MAX;
static const float irand_max = 1.0f / RAND_MAX;
float randf()
{
	uint32_t x = s_RndState;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 15;
	s_RndState = x;
	return x * imax;
	//return rand() * irand_max;
}

struct onb
{
	vec3 u;
	vec3 v;
	vec3 w;
	onb() {}
	onb(vec3& n) : w(n)
	{
		if (n.z >= -0.9999999f) // Handle the singularity
		{
			const float a = 1.0f / (1.0f + n.z);
			const float b = -n.x * n.y * a;
			u = vec3(1.0f - n.x * n.x * a, b, -n.x);
			v = vec3(b, 1.0f - n.y * n.y * a, -n.y);
			return;
		}
		//else
		//{
		u = vec3(0.0f, -1.0f, 0.0f);
		v = vec3(-1.0f, 0.0f, 0.0f);
		return;
		//}
	}
};

#define thicc 0.6f

float a = 0.4f;
vec3 ambient(a, a, a);

float depth = 0;
float chracter_width = 4.0f;
float character_height = 6.6f;
float distance_between_chracters = 2.0f;

static float minf(const float& x, const float& y)
{
	return x < y ? x : y;
}

static float maxf(const float& x, const float& y)
{
	return x > y ? x : y;
}

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
	Camera(vec3 look_from_, vec3 look_at_, int width_, int height_) : look_from(look_from_), look_at(look_at_), width(width_), height(height_)
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

class Shape
{
public:
	Shape() {}
	virtual float  Distance(const vec3& p) = 0;
};

class Sphere : public Shape
{
public:
	Sphere(vec3 c_, float r_, Material* m_) : c(c_), r(r_), m(m_) {}
	virtual float  Distance(const vec3& p)
	{
		return (p - c).length() - r;
	}

	vec3 c;
	float r;
	Material* m;
};

class Box : public Shape
{
public:
	Box(vec3 c1_, vec3 c2_) : c1(c1_), c2(c2_) {}
	Box(vec3 c1_, vec3 c2_, Material* m_) : c1(c1_), c2(c2_), m(m_) {}
	virtual float Distance(const vec3& p)
	{
		vec3 d1(p - c1);
		vec3 d2(c2 - p);

		return -minf(
			minf(
				minf(d1.x, d2.x),
				minf(d1.y, d2.y)
			),
			minf(d1.z, d2.z));
	}
	vec3 c1, c2;
	Material* m;
};

class Room : public Shape
{
public:
	Room(Shape* b1_, Shape* b2_, Shape* b3_, Material* m_) : b1(b1_), b2(b2_), b3(b3_), m(m_) {}
	virtual float Distance(const vec3& p)
	{
		
		float px = abs(p.x);
		int d_px = px / 8;
		px -= d_px * 8;

		vec3 plank_position(px, p.y, p.z);

		float distance_plank = b3->Distance(plank_position);

		return minf(-minf(b1->Distance(p), b2->Distance(p)), distance_plank);
	}

	Shape* b1;
	Shape* b2;
	Shape* b3;
	Material* m;
};

static vec3 calcNormal(const vec3& p, Shape* shp)
{
	float smallStep = 0.001f;//0.0001f;

	float x = shp->Distance(p + vec3(smallStep, 0, 0)) - shp->Distance(p + vec3(-smallStep, 0, 0));
	float y = shp->Distance(p + vec3(0, smallStep, 0)) - shp->Distance(p + vec3(0, -smallStep, 0));
	float z = shp->Distance(p + vec3(0, 0, smallStep)) - shp->Distance(p + vec3(0, 0, -smallStep));

	vec3 gradient(x, y, z);

	return gradient.norm();
}

//Arc was separate from Shape and not make a sub class of shape
//because it require a different way to compute normal
//seperate letter and box also simplify computing step
//because when computing normal of letter
//we only compute distance to letter

//Arc was used to create all the curve letter
//Arc Distance Formula
//https://www.shadertoy.com/view/wl23RK
//however this formula only apply for 2D
//so we must first project the vector "pos" (calculate by substract the point in space to the center of the curve)
//float x = pos.x * sca_cos - pos.y * sca_sin; float y = pos.x * sca_sin + pos.y * sca_cos; is the projection coordinates
struct Arc
{
	vec3 arc_center;
	float arc_tube_radius;//radius of the tube running along the arc
	float orientation_angle;
	float apature_angle;

	float sca_cos;
	float sca_sin;

	float scb_cos;
	float scb_sin;

	float ra;//radius of the circle
	float rb = 0.0f;//change this value to rb > 0.0f for interesting shapes ;)

	Arc() {}
	Arc(vec3 center, float arc_tube_radius_, float orientation_angle_, float apature_angle_, float ra_, float rb_ = 0.0f) : arc_center(center), arc_tube_radius(arc_tube_radius_), orientation_angle(orientation_angle_), apature_angle(apature_angle_), ra(ra_), rb(rb_)
	{
		orientation_angle = orientation_angle * pi / 180.0f;
		apature_angle = apature_angle * pi / 180.0f;

		sca_cos = cosf(orientation_angle);
		sca_sin = sinf(orientation_angle);

		scb_cos = cosf(apature_angle);
		scb_sin = sinf(apature_angle);
	}

	float distance(const vec3& p)
	{
		vec3 pos(p.x, p.y, depth);

		pos = (pos - arc_center);

		float x = pos.x * sca_cos - pos.y * sca_sin;
		float y = pos.x * sca_sin + pos.y * sca_cos;

		pos.x = x;
		pos.y = y;

		pos.x = abs(pos.x);

		float k = (scb_cos * pos.x > scb_sin * pos.y) ? (pos.x * scb_sin + pos.y * scb_cos) : pos.length();
		float project_distance = sqrt14(pos.dot(pos) + ra * ra - 2.0 * ra * k) - rb;

		return sqrt14(project_distance * project_distance + p.z * p.z) - arc_tube_radius;
	}
};


enum { DIFFUSE, LETTER, ARC, SUN };


static float L(const float& a, const float& b)
{
	return a < b ? a : b;
}

static float subtract(const float& d1, const float& d2)
{
	return max(-d1, d2);
}

void initialize_characters(vector<vec3>& characters, vector<Arc>& arc, Box*& surronding_bbox)
{
		
	Arc sphere_upper_arc(vec3(-10.95f, 5.27f, depth), thicc, 315.0f, 135.0f, 1.75f, depth);
	Arc sphere_lower_arc(vec3(-11.0f, 1.75f, depth), thicc, 135.0f, 135.0f, 1.75f, depth);


	arc.emplace_back(sphere_upper_arc);
	arc.emplace_back(sphere_lower_arc);

	//A
	float start_A = -7.0f;
	float end_A = start_A + chracter_width;

	vec3 A[6];
	//left leg
	A[0] = vec3(start_A, 0, depth);
	A[1] = vec3(start_A + chracter_width * 0.5f, character_height, depth);

	//A[1] = A[1] - A[0];
	//right leg
	A[2] = vec3(start_A + chracter_width, 0, depth);
	A[3] = vec3(start_A + chracter_width * 0.5f, character_height, depth);

	//A[3] = A[3] - A[2];
	//horizontal line
	A[4] = vec3(start_A + chracter_width * 0.25f, character_height * 0.25f, depth);//character_height * 0.25f
	A[5] = vec3(start_A + chracter_width * 0.75f, character_height * 0.25f, depth);//character_height * 0.25f

																				   //A[5] = A[5] - A[4];
	for (int i = 0; i < 6; ++i)
		characters.emplace_back(A[i]);

	//N
	float start_N = end_A + distance_between_chracters;
	float end_N = start_N + chracter_width;

	vec3 N[6];

	//left leg
	N[0] = vec3(start_N, 0, depth);
	N[1] = vec3(start_N, character_height, depth);

	//N[1] = N[1] - N[0];
	//right leg
	N[2] = vec3(start_N + chracter_width, 0, depth);
	N[3] = vec3(start_N + chracter_width, character_height, depth);

	//N[3] = N[3] - N[2];
	//connector
	N[4] = vec3(start_N, character_height, depth);
	N[5] = vec3(start_N + chracter_width, 0, depth);

	//N[5] = N[5] - N[4];
	for (int i = 0; i < 6; ++i)
		characters.emplace_back(N[i]);

	//G

	Arc G_arc(vec3(8.9f, 3.3f, depth), thicc, 270.0f, 130.0f, 3.3f, 0.0f);

	arc.emplace_back(G_arc);

	vec3 G[4];

	G[0] = vec3(11.0f, 0.0f, depth);
	G[1] = vec3(11.0f, 3.0f, depth);

	//G[1] = G[1] - G[0];
	//G[2] = vec3(10.0f, 3.4f, depth);
	//G[3] = vec3(12.0f, 3.4f, depth);

	G[2] = vec3(10.0f, 3.0f, depth);
	G[3] = vec3(12.0f, 3.0f, depth);

	//G[3] = G[3] - G[2];
	for (int i = 0; i < 4; ++i)
		characters.emplace_back(G[i]);

	//create a bounding box for letter
	//ray have to hit this bounding box before hitting letter
	//thus reduce computation time
	//offset was added to avoid missing point too close to ther letter
	//letter here mean all straigh line and arc
	//remove offset or set offset = 0.0f will slightly increase performance

	float offset = 0.1f;

	float left_most = -11.0f - 1.75f - thicc - offset;//S
	float right_most = 12.0f + thicc + offset;//G

	float upper_most = 5.275f + 1.75f + thicc + offset;//S
	float underware_most = 0.0f - thicc - offset;//everyone

	float front_most = depth + thicc + offset;//everyone
	float back_most = depth - thicc - offset;//everyone

	surronding_bbox = new Box(vec3(left_most, underware_most, back_most), vec3(right_most, upper_most, front_most));
}


static bool Sphere_Tracing(const Ray& r, vector<Shape*>& shapes, vector<vec3>& characters, vector<Arc>& arc, float& t, int& hit_type, int& shape_id, int& character_id, int& arc_id, Box*& surronding_bbox)
{
	float total_distance_travel = 0.0f;
	int maximum_step = 1024;

	float minimum_hit_distance = 0.0001f;
	float maximum_hit_distance = 1000.0f;


	for (int i = 0; i < maximum_step; ++i)
	{
		float mint = 1e20f;

		vec3 position(r.o + r.d * total_distance_travel);

		for (int j = 0; j < shapes.size(); ++j)
		{
			float d = shapes[j]->Distance(position);

			if (d < mint)
			{
				shape_id = j;
				mint = d;
				hit_type = DIFFUSE;
			}
		}
		float distance_to_surrounding_box = surronding_bbox->Distance(position);

		if (distance_to_surrounding_box < 5.5f)
		{
			for (int j = 0; j < characters.size(); j += 2)
			{
				vec3 begin = characters[j];
				vec3 end = characters[j + 1];

				vec3 begin_end = end - begin;

				//vec3 begin_end = characters[j + 1];

				float begin_x = -minf((begin - position).dot(begin_end) / (begin_end.length2()), 0.0f);

				begin_x = minf(begin_x, 1.0f);

				//lay vi tri hitpoint
				vec3 hit_point_letter = begin + begin_x * begin_end;

				vec3 hit_point_letter_to_eye = position - hit_point_letter; 

				float d = hit_point_letter_to_eye.length();

				
				//good distance
				d -= thicc;

				if (d < mint)
				{
					//cout << "a";
					character_id = j;
					mint = d;
					hit_type = LETTER;
				}
			}

			for (int j = 0; j < arc.size(); ++j)
			{
				float distance_s = arc[j].distance(position);
				if (distance_s < mint)
				{
					arc_id = j;
					mint = distance_s;
					hit_type = ARC;
				}
			}
		}

		total_distance_travel += mint;

		if (mint < minimum_hit_distance)
		{
			t = total_distance_travel;
			vec3 hit_point(r.o + r.d * total_distance_travel);

			float sun = 19.9f - hit_point.y;

			//if (sun <= mint)
			if (sun < 0.0f)
				hit_type = SUN;

			//t = total_distance_travel;
			//final_mint = mint;
			return true;
		}


		if (total_distance_travel > maximum_hit_distance)
		{
			return false;
		}

	}
	return false;
}


static float compute_distance_letter2(vec3& position, vector<Shape*>& shapes, vector<vec3>& characters, vector<Arc>& arc)
{
	//float sum_t = 0.0f;
	float mint = 1e20f;

	for (int j = 0; j < shapes.size(); ++j)
	{
		float d = shapes[j]->Distance(position);

		if (d < mint)
		{
			mint = d;
			//hit_id = j;
			//hit_type = DIFFUSE;
		}
	}

	for (int j = 0; j < characters.size(); j += 2)
	{
		vec3 begin = characters[j];
		vec3 end = characters[j + 1];

		vec3 begin_end = end - begin;


		float begin_x = -minf((begin - position).dot(begin_end) / (begin_end.length2()), 0.0f);


		begin_x = minf(begin_x, 1.0f);

		//lay vi tri hitpoint
		vec3 hit_point_letter = begin + begin_x * begin_end;

		vec3 hit_point_letter_to_eye = position - hit_point_letter; //o

		float d = hit_point_letter_to_eye.length();

		d -= thicc;

		if (d < mint)
			mint = d;
	}

	for (int j = 0; j < arc.size(); ++j)
	{
		float distance_s = arc[j].distance(position);
		if (distance_s < mint)
			mint = distance_s;
	}
	return mint;
}

static float compute_distance_letter_optimize(vec3& position, vector<vec3>& characters, int& character_id)
{
	vec3 begin = characters[character_id];
	vec3 end = characters[character_id + 1];

	vec3 begin_end = end - begin;

	//vec3 begin_end = characters[character_id + 1];

	float begin_x = -minf((begin - position).dot(begin_end) / (begin_end.length2()), 0.0f);


	begin_x = minf(begin_x, 1.0f);

	vec3 hit_point_letter = begin + begin_x * begin_end;

	vec3 hit_point_letter_to_eye = position - hit_point_letter; //o

																//original
	float d = hit_point_letter_to_eye.length();

	//d = powf(powf(d, 8.0f) + powf(position.z, 8.0f), 0.125f) - thicc;

	//float d = hit_point_letter_to_eye.length2();
	//float d4 = d * d;
	//float d8 = d4 * d4;
	//d = powf(d8 + powf(position.z, 8.0f), 0.125f) - thicc;

	//d = powf(powf(d, 4.0f) + powf(position.z, 8.0f), 0.125f) - thicc;

	//good distance
	d -= thicc;

	return d;
}

static float compute_distance_arc_optimize(vec3& position, vector<Arc>& arc, int& arc_id)
{
	float distance_s = arc[arc_id].distance(position);
	return distance_s;
}

static vec3 path_tracing(const Ray& r, vector<Shape*>& shapes, vector<vec3>& characters, vector<Arc>& arc, vector<Material*>& mats, Box*& surronding_bbox)
{
	Ray new_ray(r);

	vec3 L(0.0f);
	vec3 T(1.0f);
	vec3 light_direction(0.6f, 0.6f, 1.0f);
	//vec3 light_direction(0.6f, 0.9f, 1.0f);
	light_direction = light_direction.norm();


	for (int bounce = 0; bounce < 3; ++bounce)
	{
		int shape_id;
		int letter_id;
		int arc_id;
		float mint;
		float t;
		int hit_type;
		if (Sphere_Tracing(new_ray, shapes, characters, arc, t, hit_type, shape_id, letter_id, arc_id, surronding_bbox))
		{
			vec3 hit_point(new_ray.o + new_ray.d * t);

			T *= 0.2f;


			if (hit_type == DIFFUSE)
			{
				vec3 normal(calcNormal(hit_point, shapes[shape_id]));

				vec3 coord(mats[shape_id]->bsdf_sample(new_ray.d, normal));

				onb local_onb(normal);

				vec3 d(coord.x * local_onb.u + coord.y * local_onb.v + coord.z * local_onb.w);

				hit_point += normal * 0.2f;

				new_ray.o = hit_point;
				new_ray.d = d;

				float incident = d.dot(light_direction);


				//int light_id;
				float light_mint;
				float light_t;
				int light_hit_type;

				int shadow_shape_id;
				int shadow_letter_id;
				int shadow_arc_id;

				Ray light_ray(hit_point, light_direction);

				bool directional_light_sampling = Sphere_Tracing(light_ray, shapes, characters, arc, light_t, light_hit_type, shadow_shape_id, shadow_letter_id, shadow_arc_id, surronding_bbox);

				if (incident > 0 && directional_light_sampling && light_hit_type == SUN)
					L += T * vec3(500, 400, 100) * incident;
			}
			else if (hit_type == LETTER)
			{
				float smallStep = 0.001f;

				//float dx = compute_distance_letter2(hit_point + vec3(smallStep, 0.0f, 0.0f), shapes, characters, arc) - compute_distance_letter2(hit_point - vec3(smallStep, 0.0f, 0.0f), shapes, characters, arc);
				//float dy = compute_distance_letter2(hit_point + vec3(0.0f, smallStep, 0.0f), shapes, characters, arc) - compute_distance_letter2(hit_point - vec3(0.0f, smallStep, 0.0f), shapes, characters, arc);
				//float dz = compute_distance_letter2(hit_point + vec3(0.0f, 0.0f, smallStep), shapes, characters, arc) - compute_distance_letter2(hit_point - vec3(0.0f, 0.0f, smallStep), shapes, characters, arc);

				float dx = compute_distance_letter_optimize(hit_point + vec3(smallStep, 0.0f, 0.0f), characters, letter_id) - compute_distance_letter_optimize(hit_point - vec3(smallStep, 0.0f, 0.0f), characters, letter_id);
				float dy = compute_distance_letter_optimize(hit_point + vec3(0.0f, smallStep, 0.0f), characters, letter_id) - compute_distance_letter_optimize(hit_point - vec3(0.0f, smallStep, 0.0f), characters, letter_id);
				float dz = compute_distance_letter_optimize(hit_point + vec3(0.0f, 0.0f, smallStep), characters, letter_id) - compute_distance_letter_optimize(hit_point - vec3(0.0f, 0.0f, smallStep), characters, letter_id);

				vec3 normal(vec3(dx, dy, dz).norm());
				vec3 direction(new_ray.d - 2.0f * new_ray.d.dot(normal) * normal);


				hit_point += normal * 0.2f;

				onb local_onb(normal);

				new_ray.o = hit_point + normal * 0.1f;
				new_ray.d = (direction.x * local_onb.u + direction.y * local_onb.v + direction.z * local_onb.w).norm();
			}
			else if (hit_type == ARC)
			{
				float smallStep = 0.001f;

				float dx = compute_distance_arc_optimize(hit_point + vec3(smallStep, 0.0f, 0.0f), arc, arc_id) - compute_distance_arc_optimize(hit_point - vec3(smallStep, 0.0f, 0.0f), arc, arc_id);
				float dy = compute_distance_arc_optimize(hit_point + vec3(0.0f, smallStep, 0.0f), arc, arc_id) - compute_distance_arc_optimize(hit_point - vec3(0.0f, smallStep, 0.0f), arc, arc_id);
				float dz = compute_distance_arc_optimize(hit_point + vec3(0.0f, 0.0f, smallStep), arc, arc_id) - compute_distance_arc_optimize(hit_point - vec3(0.0f, 0.0f, smallStep), arc, arc_id);

				vec3 normal(vec3(dx, dy, dz).norm());
				vec3 direction(new_ray.d - 2.0f * new_ray.d.dot(normal) * normal);


				hit_point += normal * 0.2f;

				onb local_onb(normal);

				new_ray.o = hit_point + normal * 0.1f;
				new_ray.d = (direction.x * local_onb.u + direction.y * local_onb.v + direction.z * local_onb.w).norm();
			}
			else if (hit_type == SUN)
			{
				L += T * vec3(50, 80, 100);// / prev_pdf;

				return L;
			}
		}
		else
		{
			//L += T * vec3(50, 80, 100);// / prev_pdf;

			return L;
		}
	}

	return L;
}

static float Luminance(const vec3& v)
{
	return 0.2126f * v.x + 0.7152f * v.y + 0.0722 * v.z;
}

void main()
{
	int width = 960;
	int height = 540;

	int ns = 16; int step = 16;
	//int ns = 1; int step = 1;
	//int ns = 1024; int step = 16;
	float ins = 1.0f / ns;

	float iWidth = 1.0f / width;
	float iHeight = 1.0f / height;

	
	Camera cam(vec3(0, 1.0, 18.0f), vec3(0, 1.0, 0), width, height);

	vector<vec3> c;
	c.resize(width * height);

	vector<Shape*> shapes;

	//shapes.resize(1);

	Material* diffuse_room = new Diffuse(vec3(0.95f));

	//original
	/*
	Shape* lower_room = new Box(vec3(-30.0f, -0.5f, -30.0f), vec3(30.0f, 18.0f, 30.0f), diffuse_room);
	Shape* upper_room = new Box(vec3(-25.0f, 17.0f, -25.0f), vec3(25.0f, 20.0f, 25.0f), diffuse_room);
	Shape* plank = new Box(vec3(1.5f, 18.5f, -25.0f), vec3(6.5f, 20.0f, 25.0f), diffuse_room);
	*/

	/*Shape* lower_room = new Box(vec3(-30.0f, -0.5f, -30.0f), vec3(30.0f, 30.0f, 30.0f), diffuse_room);
	Shape* upper_room = new Box(vec3(-25.0f, 19.0f, -25.0f), vec3(25.0f, 22.0f, 25.0f), diffuse_room);
	Shape* plank = new Box(vec3(1.5f, 20.5f, -25.0f), vec3(6.5f, 22.0f, 25.0f), diffuse_room);
	*/

	Shape* lower_room = new Box(vec3(-40.0f, -0.6f, -30.0f), vec3(40.0f, 18.0f, 30.0f), diffuse_room);

	Shape* upper_room = new Box(vec3(-35.0f, 17.0f, -25.0f), vec3(35.0f, 20.0f, 25.0f), diffuse_room);

	Shape* plank = new Box(vec3(1.5f, 18.5f, -25.0f), vec3(6.5f, 20.0f, 25.0f), diffuse_room);

	Shape* room = new Room(lower_room, upper_room, plank, diffuse_room);


	vector<Material*> mats;

	mats.emplace_back(diffuse_room);

	shapes.emplace_back(room);

	int num_primitives = shapes.size();

	vector<vec3> characters;
	vector<Arc> arces;

	Box* surronding_bbox;

	initialize_characters(characters, arces, surronding_bbox);

	omp_set_num_threads(128);

	clock_t t_render = clock();

	for (int j = 0; j < height; ++j)
	{
		fprintf(stderr, "\rRendering: (%d spp) %5.2f%%", ns, 100.0f * j / (height - 1));
#pragma omp parallel for schedule(guided)
		for (int i = 0; i < width; ++i)
		{
			vec3 sum(0.0f);
			int num_sample_used = 0;

			bool converge = false;

			float convergence_rate = 0.0001f;

			float sum_square = 0.0f;
			float sum_so_far = 0.0f;


			for (int s = 0; s < ns; s += step)
			{
				for (int num = 0; num < step; ++num)
				{
					float p = ((float)i + randf()) * iWidth;
					float q = ((float)j + randf()) * iHeight;

					Ray r = cam.generate_ray(p, q);

					vec3 color = path_tracing(r, shapes, characters, arces, mats, surronding_bbox);
					sum += color;

					float lux = Luminance(color);

					sum_square += lux * lux;
					sum_so_far += lux;
				}
				num_sample_used += step;

				float mean_sum = sum_so_far / num_sample_used;
				float variance = (sum_square / num_sample_used - mean_sum * mean_sum) / (num_sample_used - 1);

				vec3 value(sum / num_sample_used);

				if (variance < convergence_rate && value.minc() > 0.2f)
				{
					c[j * width + i] = value;
					converge = true;
					break;
				}
			}
			if (!converge)
				c[j * width + i] = sum * ins;
		}
	}

	string s = "a_a_a_a_a_Sang_1024.ppm";

	std::ofstream ofs(s, std::ios::out | std::ios::binary);

	ofs << "P3\n" << width << " " << height << "\n255\n";

	int size = width * height;

	for (int i = 0; i < size; ++i)
	{
		vec3 color = c[i] + 14.0f / 241.0f;
		vec3 o = color + 1.0f;//vec3(1.0f);
		color = vec3(color.x / o.x, color.y / o.y, color.z / o.z) * 255.0f;

		ofs << (int)color.x << " " << (int)color.y << " " << (int)color.z << "\n";
	}

	t_render = clock() - t_render;


	std::cout << "\nRendering time: " << ((double)t_render) / CLOCKS_PER_SEC << "\n";
	ofstream log_file("log.txt");
	log_file << ((double)t_render) / CLOCKS_PER_SEC << " s";
	vector<vec3>().swap(c);
	ofs.clear();
	log_file.clear();

	//getchar();

}
