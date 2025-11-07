#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

vector<int> idx;
vector<double> x,y,z,vx,vy,vz,mass;

void read_data(int n)
{
    fstream fin;
    fin.open("../data/data_"+ to_string(n) +".csv",ios::in);

    string line, value;
    getline(fin, line);

    for(int i =0 ;i <n ; i++)
    {
        getline(fin, line);

        stringstream s(line);

        for(int _ =0 ; _< 8 ; _++)
        {
            getline(s,value, ',');
            if(_==0) idx[i] = stoi(value);
            else if(_==1) x[i] = stod(value);
            else if(_==2) y[i] = stod(value);
            else if(_==3) z[i] = stod(value);
            else if(_==4) vx[i] = stod(value);
            else if(_==5) vy[i] = stod(value);
            else if(_==6) vz[i] = stod(value);
            else if(_==7) mass[i] = stod(value);
        }
        
    }
    
    fin.close();
}

void allocate_memory(int n)
{
    idx = vector<int>(n);
    x = vector<double>(n);
    y = vector<double>(n);
    z = vector<double>(n);
    vx = vector<double>(n);
    vy = vector<double>(n);
    vz = vector<double>(n);
    mass = vector<double>(n);
}

void free_memory()
{
    idx.clear();
    x.clear();
    y.clear();
    z.clear();
    vx.clear();
    vy.clear();
    vz.clear();
    mass.clear();
}

void show_data(int n)
{
    for(int i =0 ; i<n ; i++)
    {
        cout << idx[i] << " " << x[i] << " " << y[i] << " " << z[i] << " "
            << vx[i] << " " << vy[i] << " " << vz[i] << " " << mass[i] << endl;
    }
}

int main()
{
    int n = 10;
    allocate_memory(n);

    read_data(n);

    show_data(n);
    
    free_memory();
}