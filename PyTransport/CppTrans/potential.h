//#This file is part of PyTransport.

//#PyTransport is free software: you can redistribute it and/or modify
//#it under the terms of the GNU General Public License as published by
//#the Free Software Foundation, either version 3 of the License, or
//#(at your option) any later version.

//#PyTransport is distributed in the hope that it will be useful,
//#but WITHOUT ANY WARRANTY; without even the implied warranty of
//#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#GNU General Public License for more details.

//#You should have received a copy of the GNU General Public License
//#along with PyTransport.  If not, see <http://www.gnu.org/licenses/>.

// This file contains a prototype of the potential.h file of PyTransport -- it is edited by the PyTransScripts module

#ifndef POTENTIAL_H  // Prevents the class being re-defined
#define POTENTIAL_H


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

// #Rewrite
// Potential file rewriten at Fri Mar 17 15:11:14 2023

class potential
{
private:
	int nF; // field number
	int nP; // params number which definFs potential
    
    
public:
	// flow constructor
	potential()
	{
// #FP
nF=2;
nP=0;

//        p.resize(nP);
        
// pdef

    }
	
    //void setP(vector<double> pin){
    //    p=pin;
    //}
	//calculates V()
	double V(vector<double> f, vector<double> p)
	{
		double sum ;
        
// Pot
  sum=std::pow(f[0], 2) - 0.66666666666666663/std::pow(f[1], 2);
         return sum;
	}
	
	//calculates V'()
	vector<double> dV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF,0.0);
	
// dPot

 sum[0]=2*f[0];

 sum[1]=1.3333333333333333/std::pow(f[1], 3);
        
		return sum;
	}
    
	// calculates V''
	vector<double> dVV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF*nF,0.0);
		
// ddPot
  double x0 = -2*f[0]/f[1];

 sum[0]=2 + 1.3333333333333333/std::pow(f[1], 2);

 sum[2]=x0;

 sum[1]=x0;

 sum[3]=-4.0/std::pow(f[1], 4);
     
        return sum;
	}
    
	// calculates V'''
	vector<double> dVVV(vector<double> f, vector<double> p)
	{
        vector<double> sum(nF*nF*nF,0.0);
// dddPot
  double x0 = 4*f[0];
  double x1 = 4/f[1];
  double x2 = std::pow(f[1], -3);
  double x3 = -x1 - 5.333333333333333*x2;
  double x4 = x0/std::pow(f[1], 2);

 sum[0]=-x0;

 sum[4]=-x1 - 5.3333333333333339*x2;

 sum[2]=x3;

 sum[6]=x4;

 sum[1]=x3;

 sum[5]=x4;

 sum[3]=x4;

 sum[7]=16.0/std::pow(f[1], 5);
       
        return sum;
	}
    
    int getnF()
    {
        return nF;
    }
    
    int getnP()
    {
        return nP;
    }

};
#endif