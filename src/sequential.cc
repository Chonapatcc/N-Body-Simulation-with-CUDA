#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
using namespace std;

const int n = 10000;
const double N_steps = 10;
const double dt = 0.1;
const double G = 6.6743e-11;
const double softening = 1e-14;

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

    void update_position(double dt)
    {
        x += vx*dt;
        y += vy*dt;
        z += vz*dt;
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
    particles.resize(n);
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
    fout.open("../result/sequential_force_" + to_string(n) +".csv", ios::out);

    for(int i =0; i<n ; i++)
    {
        fout << particles[i].idx << "," << particles[i].force << endl;
    }

    fout.close();
}


void nbody()
{
    for(int i =0 ; i<n ; i++)
    {
        double ax=0,ay=0,az=0;

        for(int j=0 ;j<n;j++)
        {
            if(i==j) continue;
            
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;

            double dist_sqr = dx*dx + dy*dy + dz*dz + softening*softening;
            
            double invDist = 1.0/ sqrtf(dist_sqr);
            double invDistCube = invDist * invDist * invDist;
            
            double magnitude = G * particles[j].mass * invDistCube;

            ax += dx * magnitude;
            ay += dy * magnitude;
            az += dz * magnitude;

            double force = magnitude * particles[i].mass;
            particles[i].force += force;
        }
        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
        particles[i].vz += az * dt;

        particles[i].update_position(dt);
    }
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

    cout << "Time elapsed: " << elapsed.count() << " seconds" <<endl;
    save_force();
    
}