#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		
#include <GL/freeglut.h>	
#endif

const unsigned int screenWidth = 600, screenHeight = 600;

struct vec3 {
	float x, y, z;
	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float d) const { return vec3(x / d, y / d, z / d); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	void operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator-() const { return vec3(-x, -y, -z); }
	vec3 normalize() const { return (*this) * (1 / (Length() + 0.000001)); }
	float Length() const { return sqrtf(x * x + y * y + z * z); }
	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

class Material {
	vec3 kd, ks;
	float  shininess;
public:
	vec3 ka;
	Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) : ka(_ka), kd(_kd), ks(_kd) { shininess = _shininess; }
	vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad) {
		vec3 reflRad(0, 0, 0);
		float cosTheta = dot(normal, lightDir);
		if (cosTheta < 0) return reflRad;
		reflRad = inRad * kd * cosTheta;
		vec3 halfway = (viewDir + lightDir).normalize();
		float cosDelta = dot(normal, halfway);
		if (cosDelta < 0) return reflRad;
		return reflRad + inRad * ks * pow(cosDelta, shininess);
	}
};

struct Hit {
	float t;
	vec3 position;
	vec3 normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = _dir.normalize(); }
};


class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float b = dot(dist, ray.dir) * 2.0;
		float a = dot(ray.dir, ray.dir);
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0 && t2 > 0)       hit.t = t2;
		else if (t2 <= 0 && t1 > 0)  hit.t = t1;
		else if (t1 < t2)            hit.t = t1;
		else                         hit.t = t2;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = w.Length();
		right = cross(vup, w).normalize() * f * tan(fov / 2);
		up = cross(w, right).normalize() * f * tan(fov / 2);
	}
	Ray getray(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / screenWidth - 1) + up * (2.0 * (Y + 0.5) / screenHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = _direction.normalize();
		Le = _Le;
	}
};

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 3, 3)));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(10, 10, 10);	
		vec3 ka = kd * M_PI;
		objects.push_back(new Sphere(vec3(0, 0, 0), 0.5f, new Material(ka, kd, ks, 50)));
	}

	void render( vec3 image[] ) {
#pragma omp parallel for
		for (int Y = 0; Y < screenHeight; Y++) {
			for (int X = 0; X < screenWidth; X++) image[Y * screenWidth + X] = trace(camera.getray(X, Y));
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); 
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			outRadiance += hit.material->shade(hit.normal, -ray.dir, light->direction, light->Le);
		}
		return outRadiance;
	}
};

Scene scene;

void getErrorInfo(unsigned int handle) {
	int logLen, written;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	
	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			
	out vec4 fragmentColor;		

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	
public:
	void Create(vec3 image[]) {
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1};	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     

		
		glGenTextures(1, &textureId);  				
		glBindTexture(GL_TEXTURE_2D, textureId);    
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, image); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 image[screenWidth * screenHeight];	

void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);
	scene.build();
	scene.render(image);	
	fullScreenTexturedQuad.Create(image);

	
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	
	shaderProgram = glCreateProgram();
	if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	

	
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	glUseProgram(shaderProgram); 	
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  
	}
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); 
}

int main(int argc, char * argv[]) {
	int majorVersion = 3, minorVersion = 3; 	
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(screenWidth, screenHeight);				
	glutInitWindowPosition(100, 100);							
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

