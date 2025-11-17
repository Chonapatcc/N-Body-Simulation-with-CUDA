#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;

const int n = 10000;
const double theta = 0.5;
const double N_steps = 10;
const double dt = 0.1;
const double G = 6.6743e-11;
const double softening = 1e-8;

struct Particle
{
    int idx;
    double x,y,z;
    double vx,vy,vz;
    double mass;
    double force;

    Particle() {}

    Particle(int idx, double x, double y, double z, double vx, double vy, double vz, double mass)
        : idx(idx), x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), mass(mass), force(0) {}

    
    __host__ __device__    
    void update_position(double dt)
    {
        x += vx*dt;
        y += vy*dt;
        z += vz*dt;
    }
};

struct Tree
{
    double COM_x, COM_y, COM_z;
    double mass;
    Particle* particle;
    bool is_leaf;
    bool has_particle;
    double size;
    Tree* child_particles[8];

    Tree(){}

    Tree() : COM_x(0), COM_y(0), COM_z(0), mass(0), particle(nullptr), is_leaf(false), has_particle(false), size(0)
    {
        for(int i=0 ; i<8;i++)
        {
            child_particles[i] = nullptr;
        }
    }

    bool isEmpty()
    {
        return !has_particle;
    }
};

vector<Particle> particles;

void read_data()
{
    fstream fin;
    fin.open("../data/data_"+ to_string(n) +".csv",ios::in);

    string line, value;
    getline(fin, line);

    for(int i =0 ;i <n ; i++)
    {
        getline(fin, line);

        stringstream s(line);
        
        int idx;
        double x, y, z, vx,vy,vz,mass;

        for(int _ =0 ; _< 8 ; _++)
        {
            getline(s,value, ',');
            if(_==0) idx = stoi(value);
            else if(_==1) x = stod(value);
            else if(_==2) y = stod(value);
            else if(_==3) z = stod(value);
            else if(_==4) vx = stod(value);
            else if(_==5) vy = stod(value);
            else if(_==6) vz = stod(value);
            else if(_==7) mass = stod(value);
        }
        particles[i] = Particle(idx, x, y, z, vx, vy, vz, mass);
    }
    
    fin.close();
}

void allocate_memory()
{
    particles = vector<Particle>(n);
}

void show_data()
{
    for(int i =0 ; i<n ; i++)
    {
        cout << particles[i].idx << " "
             << particles[i].x << " "
             << particles[i].y << " "
             << particles[i].z << " "
             << particles[i].vx << " "
             << particles[i].vy << " "
             << particles[i].vz << " "
             << particles[i].mass << " " <<endl;
    }
}

void show_force()
{
    for(int i =0 ;i<n ; i++)
    {
        cout << "particle " << particles[i].idx << " force: " << particles[i].force << endl;
    }
}

void save_force()
{
    fstream fout;
    fout.open("../result/sequential_best_force_" + to_string(n) +".csv", ios::out);

    for(int i =0; i<n ; i++)
    {
        fout << particles[i].idx << "," << particles[i].force << endl;
    }

    fout.close();
}



vector<double> calculate_COM(vector<Particle> &particle_list)
{
    double total_mass =0;
    double com_x =0 , com_y =0 , com_z =0 ;
    for(Particle p : particle_list)
    {
        total_mass += p.mass;
        com_x += p.x * p.mass;
        com_y += p.y * p.mass;
        com_z += p.z * p.mass;
    }
    com_x /= total_mass;
    com_y /= total_mass;
    com_z /= total_mass;
    return {com_x, com_y, com_z, total_mass};
}

vector<double> calculate_size(vector<Particle> &particle_list , double center_x , double center_y, double center_z)
{
    double max_x = 0 , max_y =0, max_z =0 ;
    for(Particle p : particle_list)
    {
        double dx = fabs(p.x - center_x);
        double dy = fabs(p.y - center_y);
        double dz = fabs(p.z - center_z);
        if(dx > max_x) max_x = dx;
        if(dy > max_y) max_y = dy;
        if(dz > max_z) max_z = dz;
    }
    return {max_x, max_y, max_z};
}

void build_tree(Tree* tree, vector<Particle> &particle_list)
{
    if (particle_list.size() == 0 )
    {
        tree->is_leaf = true;
        tree->has_particle = false;
        tree->particle = nullptr;
        return;
    }
    else if(particle_list.size() == 1)
    {
        tree->is_leaf = true;
        tree->has_particle = true;
        tree->particle = &particle_list[0];
        tree->mass = particle_list[0].mass;
        tree->COM_x = particle_list[0].x;
        tree->COM_y = particle_list[0].y;
        tree->COM_z = particle_list[0].z;
        return;
    }
    else
    {
        tree->is_leaf = false;
        tree->has_particle = true;
        vector<double> com = calculate_COM(particle_list);
        tree->COM_x = com[0];
        tree->COM_y = com[1];
        tree->COM_z = com[2];
        tree->mass = com[3];

        vector<double> sizes = calculate_size(particle_list, tree->COM_x, tree->COM_y, tree->COM_z);
        double child_size = max(sizes[0],max(sizes[1],sizes[2]));
        tree->size = child_size * 2;
        vector<vector<Particle>> child_particles(8);

        for(Particle p : particle_list)
        {
            int octant = 0;
            if(p.x > tree->COM_x) octant |= 1;
            if(p.y > tree->COM_y) octant |= 2;
            if(p.z > tree->COM_z) octant |= 4;

            child_particles[octant].push_back(p);
        }
        for(int i =0 ;i <8; i++)
        {
            double offset_x = ((i & 1) ? 0.5 : -0.5) * child_size;
            double offset_y = ((i & 2) ? 0.5 : -0.5) * child_size;
            double offset_z = ((i & 4) ? 0.5 : -0.5) * child_size;
            tree->child_particles[i] = new Tree();
            build_tree(tree->child_particles[i], child_particles[i]);
        }

    }
}

double calculate_distance(Particle &p, Tree* tree)
{
    double dx = tree->COM_x - p.x;
    double dy = tree->COM_y - p.y;
    double dz = tree->COM_z - p.z;

    return sqrtf(dx*dx + dy*dy + dz*dz);
}

void calculate_acceleration(Particle &p, Tree* tree, double& ax, double& ay, double& az)
{
    if(tree==nullptr || tree->mass == 0) return;
    if(tree->is_leaf == true && tree->particle == &p) return;

    double d = calculate_distance(p,tree);
    
    if((tree->size / d) < theta || tree->is_leaf == true)
    {
        double dx = tree->COM_x - p.x;
        double dy = tree->COM_y - p.y;
        double dz = tree->COM_z - p.z;
        double dist_sqr = dx*dx + dy*dy + dz*dz + softening*softening;
                
        double invDist = 1.0/ sqrtf(dist_sqr);
        double invDistCube = invDist * invDist * invDist;
        
        double magnitude = G * tree->mass * invDistCube;

        ax += dx * magnitude;
        ay += dy * magnitude;
        az += dz * magnitude;
    }
    else
    {
        for(int i =0 ;i<8;i++)
        {
            if(tree->child_particles[i] != nullptr)
            {
                calculate_acceleration(p, tree->child_particles[i],ax,ay,az);
            }
        }
    }

}

void nbody()
{
    Tree* root = new Tree();
    build_tree(root,particles);
    
    double theta = 0.5;
    for(int i =0 ; i<n ; i++)
    {
        double ax=0,ay=0,az=0;

        calculate_acceleration(particles[i], root, ax, ay, az);

        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
        particles[i].vz += az * dt;

        double force = sqrtf(ax*ax + ay*ay + az*az) * particles[i].mass;
        particles[i].force = force;
   
    }
    for(int i=0;i<n;i++)
    {
        particles[i].update_position(dt);
    }

    delete root;
}

void save_time(double time_elapsed)
{
    fstream fout;
    fout.open("../result/sequential_best_time_" + to_string(n) +".txt", ios::out);
    fout << "Time elapsed: " << time_elapsed << " seconds" <<endl;
    fout.close();
}

int main()
{
    allocate_memory();

    read_data();
    
    auto start = chrono::high_resolution_clock::now();
    for(int step =0 ; step< N_steps;step++)
    {
        nbody();
    }
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    save_time(elapsed.count());
    save_force();
}