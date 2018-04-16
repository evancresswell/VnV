#include <iostream> 
#include <algorithm> 
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <float.h>
#include <vector>
using namespace std;

int ghost_num;
int nx;
int a_len;
double t_final;
double t_start;
double t;
int inter_start;
int inter_end;
double vel;

// initialize spatiall domain
double right_boundary;
double left_boundary;

//----------------------------------------------------------------------
double a6j(double l_face,double r_face,double node_val)
{
	double val;
	val = 6.* ( node_val - .5* ( l_face + r_face  )   ) ;
	return val;
}

//--------------------Interpolation--------------------------------------------------//
//----------------------------------------------------------------------
double interp_linear(double left,double right,double u)
{
	//calculated line through left and right points then returns value at left + .5*dx 
	//ASSUMES UNIFORM DX
	double mem_val;

	mem_val = (left+right ) / 2.;
	if(u>0)
		return mem_val;
	if(u<0)
		return -mem_val;
}

//----------------------------------------------------------------------
void printFlux(vector<double> fl, vector<double> fr)
{
		cout << "Flux has length " << fl.size() << "\n";
		// 	print membrane fluxes 
		if(1)
		{	
				cout << "fl = [ ";
				for(int i=0 ; i < a_len-1 ; i++)
					cout << fl[i]<<" ";
				cout << " ]\n\n";
				cout << "fr = [ ";
				for(int i=0 ; i < a_len-1 ; i++)
					cout << fr[i]<<" ";
				cout << " ]\n\n";
		}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
double interp_parabolic(double x, double left, double middle , double right, double left_x, double middle_x, double right_x, double u)
{
	//calculated line through left and right points then returns value at left + .5*dx 
	double val;
	double coef1,coef2,coef3;
	coef1 = ( (x - middle_x) * (x - right_x) ) / ( (left_x - middle_x) * (left_x - right_x) );
	coef2 = ( (x - left_x) * (x - right_x) ) / ( (middle_x - left_x) * (middle_x - right_x) );
	coef1 = ( (x - left_x) * (x - middle_x) ) / ( (right_x - left_x) * (right_x - middle_x) );
	
	val = coef1 * left + coef2 * middle + coef3 * right ;

	return val;
}

//-------------------------Trapezoid rule-------------------------------------------
double trap(double dx, double ly, double lr)
{
	// use trapezoid rule to integrate between lb and rb
	double val;
	val = ( dx*( ly+lr ) )/2. ;
	
	return val;
}


//--------------------Membrane Fluxes--------------------------------------------------//
//----------------------------------------------------------------------
double membrane_flux(double left,double right,double u)
{
	double flux;
	if(u>0)
		flux =  left;
	if(u<0)
		flux = right;
	return flux;
}

//----------------------------------------------------------------------
double membrane_flux_linear(double left,double right,double u)
{
	double flux;
	flux = interp_linear(left, right, u);
	return flux;
}

//----------------------------------------------------------------------
double membrane_flux_parabolic(double x, double y1, double y2, double y3, double x1, double x2, double x3, double u)
{
	double flux;
	flux = interp_parabolic(x, y1, y2, y3, x1, x2, x3, u);
	return flux;
}

//----------------------------------------------------------------------
double L2error(double x[], double a[], double t, int len_a, double u)
{
	double L2_error=0;
	for(int i=ghost_num;i<len_a-ghost_num;i++)
	{
		L2_error += pow(a[i] - sin(x[i]+u*t), 2.);
		cout << pow(a[i] - sin(x[i]+u*t), 2.) << "\n";
	}
	L2_error = pow(L2_error,.5);
	L2_error /= (len_a-ghost_num*2);
	return L2_error;
}

//*
void writeError(vector<double> &error, int len_a, string output,double dt)
{
	//print solutions
	string filename = output + ".out";
	ofstream output1(filename.c_str(), fstream::app);
	output1 << dt << "\n";
	for (int i=0; i < len_a; i++) 
		output1 << error[i] << "\n";
	output1 << "\n";
	output1.close();
}
//*/

//----------------------------------------------------------------------

double smallest(double x, double y, double z) {

  double smallest = DBL_MAX;

  if (x < smallest)
    smallest=x;
  if (y < smallest)
    smallest=y;
  if(z < smallest)
    smallest=z;

  return smallest;
}

//-------------------------Monotonize 1-----------------------------------
// first order reconstrunction
//calculates single slope for given node
double daj1(double a[], int a_len, int j )
{
	// calculate the average slope in the jth zone
	double daj_m;
	double daj;
	daj =  (a[j+1] - a[j] + a[j] - a[j-1] )/2.;

	if( ((a[j+1] - a[j]) * (a[j] - a[j-1])) > 0.)
		daj_m = smallest( fabs(daj) , 2.*fabs(a[j]-a[j-1]), (daj/fabs(daj))*2.*fabs(a[j]-a[j-1]) );
	else
		daj_m = 0;
	return daj_m;
}

//-------------------------Monotonize 2-----------------------------------
//----------------------------------------------------------------------
// second order reconstrunction
//calculates single slope for given node
double daj2(double a[], int a_len, int j )
{
	// calculate the average slope in the jth zone
	double daj_m;
	double daj;
	daj = (1./3.) * ( (3./2.) * (a[j+1] - a[j]) + (3./2.) * (a[j] - a[j-1]) );

	//--- monotonization ---//
	//if( ((a[j+1] - a[j]) * (a[j] - a[j-1])) > 0.)
//		daj_m = smallest( fabs(daj) , 2.*fabs(a[j]-a[j-1]), (daj/fabs(daj))*2.*fabs(a[j]-a[j-1]) );
//	else
//		daj_m = 0;
//	return daj_m;
	//----------------------//

	return daj;
}

//----------------------------------------------------------------------
//---------------------Reconstruction--------------------------------------------------//
void reconV(double a[], double x[], int a_len, double dt, double v,	vector<double> & l_face, vector<double> & r_face, vector<double> & vdaj)
{
	// need to initialize vdaj in main and pass in EMPTY VECTOR
	for(int i=1; i<=a_len-1; i++)
		vdaj.push_back( daj1(a, a_len, i) ); //calculate daj for each point i=j
	
	for(int i=ghost_num; i<=a_len-ghost_num; i++)
	{
		l_face.push_back( a[i]-vdaj[i-ghost_num]/2.);
		r_face.push_back( a[i]+vdaj[i-ghost_num]/2.);		
	}
}
//----------------------------------------------------------------------
//---------------------Flux calculation--------------------------------------------------//
double getFlux1stOrder(double a[], double x[], int a_len, double dt, double dx, double v, vector<double> l_face,	vector<double> r_face, vector<double> vdaj, vector<double> & fl, vector<double> & fr)
{
	// calculate domain of dependence
	// ASSUMING CONSTANT DX
	// v is scalar and dt is static
	// only going to work for 2nd order: need better integration for 3rd order.......

	// flux vector at the left/right face
	for(int i=ghost_num; i<=a_len-ghost_num; i++)
	{
		// -----------------------Left Flux---------------------//	
		fl.push_back(a[i-1]);
		// ------------------------------------------------------//	

		// -----------------------Right Flux---------------------//	
		fr.push_back(a[i]);
		// ------------------------------------------------------//	
	}
}
//----------------------------------------------------------------------
//---------------------Averaging--------------------------------------------------//
double getFlux2ndOrder(double a[], double x[], int a_len, double dt, double dx, double v, vector<double> l_face,	vector<double> r_face, vector<double> vdaj, vector<double> & fl, vector<double> & fr)
{
	// calculate domain of dependence
	// ASSUMING CONSTANT DX
	// v is scalar and dt is static
	// only going to work for 2nd order: need better integration for 3rd order.......

	double y = v*dt;
	double al;
	double ar;
	double m;
	double b;
	double xl;
	double xr;
	double dx_trap;

	cout << "daj has lenth " << vdaj.size() << "\n";
	// flux vector at the left/right face
	for(int i=ghost_num; i<=a_len-ghost_num; i++)
	{
		// -----------------------Left Flux---------------------//	
		// calculate slope and intercep to get line between j-1/2 and j-1/2 - y
		m = vdaj[i-ghost_num]; // vdaj for a[ghost_num]  is vdaj[0] 
		b = a[i-1]-x[i-1]*m;
		// set x values for right and left of integration boundary
		xl = x[i]- (dx/2.) - y ;
		xr = x[i] - (dx/2.);
		al = m*xl+b;
		ar = l_face[i-ghost_num];
		dx_trap = fabs(xl-xr);
		
		fl.push_back(trap(dx_trap, al, ar)/y);
		// ------------------------------------------------------//	

		// -----------------------Right Flux---------------------//	
		// calculate slope and intercep to get line between j+1/2 and j+1/2 - y
		m = vdaj[i-ghost_num+1]; // vdaj for a[1] is vdaj[0]
		b = a[i]-x[i]*m;
		// set x values for right and left of integration boundary
		xl = x[i]+ (dx/2.) - y ;
		xr = x[i] + (dx/2.) ;
		al = m*xl+b;
		ar = r_face[i - ghost_num];
		dx_trap = fabs(xl-xr);
		
		fr.push_back(trap(dx_trap, al, ar)/y);
		// ------------------------------------------------------//	
	}
}

//----------------------------------------------------------------------
double flux3rdOrderL(double a[],double x[], double vdaj[], double l_face[], double r_face[], double a6j, int j)
{
	// Calculated 3rd order flux at right membrane of jth node
	// Only works for postive velocity!
	if(vel<0)
	{
		cout << "need to use upwinding from the right direction!!";
		exit(0);
	}

	double arj=r_face[j];
	double alj=l_face[j];
	double val;
	val =  arj - (x[j+ghost_num]/2.) * ( arj - alj * (1 - (2./3.)*x[j+ghost_num])*a6j )  ;
	return val;
}
//----------------------------------------------------------------------


//----------------------------------------------------------------------
//---------------------advect one step--------------------------------------------------//
vector<double> advectV(double a[], double dt, double dx, double v, vector<double> fl, vector<double> fr)
{
	// Given that the fluxes have been calculated for the correct membrane (right or left) we simply upwind... 
	// Therefore v does not need to be considered.
	double sum=0;
	vector<double> temp;
	for(int i=1; i<=a_len-ghost_num; i++)
	{
		temp.push_back( a[i] + v * (dt/dx) * (fl[i-ghost_num] - fr[i-ghost_num+1]) ); //flux at the left and right membrane of the cell
		sum =+ (fl[i-ghost_num] - fl[i-ghost_num+1]);
	}

	cout << "Sum of TEMP vector is: " << sum << "\n";
	return temp;
}
//----------------------------------------------------------------------

//---------------------advect one step--------------------------------------------------//
double advect(double a[], double dt, double dx, double v, double fl[], double fr[],int i)
{
	// Given that the fluxes have been calculated for the correct membrane (right or left) we simply upwind... 
	// Therefore v does not need to be considered.
	double temp;
	int ghost_num = 2;
	temp=  a[i] + v * (dt/dx) * (fl[i-ghost_num] - fr[i-ghost_num]) ; //flux at the left and right membrane of the cell
	cout << "RJL " <<a[i]<< " " << fl[i-ghost_num] << " " << fr[i-ghost_num] << "\n"; 

	return temp;
}
//----------------------------------------------------------------------


//---------------------Write Solution--------------------------------------------------//
void writeSolution(double x[], double t, double mass, int len_a, double t_start)
{
	//Printing to file format
	//	t
	//	mass
	//	x[1] x[2] ... x[len_a]
	//

	//print solutions
	ofstream output1("scalar_advec_sol.out", fstream::app);
	for (int i=0; i < len_a; i++) 
		output1 << x[i] << " ";
	output1 << "\n";
	output1.close();

	//print times
	ofstream output2("scalar_advec_t.out", fstream::app);
	output2 << t << "\n";
	output2.close();

	//print mass
	ofstream output3("scalar_advec_mass.out", fstream::app);
	output3 << mass << "\n";
	output3.close();
}

double totalMass(double x[],int a_len)
{
	double mass = 0;
	for(int i=2 ; i<=a_len-2 ; i++)	
		mass += x[i];
	return mass;
}



//---------------------1st order Solver--------------------------------------------------//
// with monotonizing and averaging
void solve1stOrder(double a[], double x[], double dt, double dx, double v, string output)
{
	/*
	vector<double> l_face;
	vector<double> r_face;
	vector<double> vdaj;
	vector<double> fl;
	vector<double> fr;
	vector<double> temp;
	*/
	double l_face[nx];
	double r_face[nx];
	double vdaj[a_len-2];
	double fl[nx];
	double fr[nx];
	double temp[a_len];

	// set index variables
	inter_start = ghost_num;
	inter_end = a_len-ghost_num;
	double mass;
	double fsum;
	double sum;
	int n=0;
	vector<double> l2_error;
	t = 0;
	ghost_num = 2; //SHOULD BE 2
	
	while (t<t_final)
	{
		// clear vectors for new timestep	
		/*
		vdaj.clear();
		fl.clear();
		fr.clear();
		temp.clear();
		l_face.clear();
		r_face.clear();
		*/

		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";
		
		// 	set ghost cells
		cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";


		// write solution to file
		cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		//-----------------------------------forward step----------------------------------//
		//---------------------------------------------------------------------------------//


		//--------------- reconstruction ------------------//
		cout << "Reconstructing profile\n";
		// reconstruction using <double> vector s
		//recon(a, x, a_len, dt, v, l_face, r_face, vdaj);

		// need to initialize vdaj in main and pass in EMPTY VECTOR
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj1(a, a_len, i) ; //calculate daj for each point i=j
	
		for(int i=inter_start; i<=inter_end; i++)
		{
			l_face[i-ghost_num] =  a[i]-vdaj[i-ghost_num]/2.;
			r_face[i-ghost_num] =  a[i]+vdaj[i-ghost_num]/2.;		
		}
		//-------------------------------------------------//
			
		// ---------------flux calculation ----------------------//
		cout << "Calculating Flux\n";
		for(int i=0; i<nx; i++)
			fl[i] = a[i+ghost_num-1];

		for(int i=0; i<nx; i++)
			fr[i] = a[i+ghost_num];
		// ------------------------------------------------------//	

		//use <double> vectors
		//fsum =  getFlux2ndOrder(a, x, a_len, dt, dx, v, l_face,	r_face, vdaj,fl,fr);
		//fsum =  getFlux1stOrder(a, x, a_len, dt, dx, v, l_face,	r_face, vdaj,fl,fr);

		cout << "Update\n\n";
		fsum = 0;

		cout << "fl = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fl[i]<<" ";
		cout << " ]\n\n";

		cout << "fr = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fr[i]<<" ";
		cout << " ]\n\n";


		// save forward step to temporary variable
		for(int i=inter_start;i<=inter_end-1; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}
		//temp = advectV(a, dt, dx, v, fl, fr);


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		l2_error.push_back(L2error(x, a, t, a_len, v));
		for(int i=ghost_num ; i <= a_len-ghost_num ; i++)
		{
			a[i] = temp[i];
		}

		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		// debug printout	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n";
		
		mass = totalMass(a, a_len);
		cout << "Total mass is: " << mass << "\n";
		
		n++;
		cout << "\n";
		cout<< "//----------------------------------------------------//\n";	

		//-------------------------------------------------//
		// Write L2 error
		writeError(l2_error, a_len, output, dt);

	}
	//end of while loop
}
//----------------------------------------------------------------------

//---------------------2nd order solveSolver--------------------------------------------------//
// with monotonizing and averaging
void solve2ndOrder(double a[], double x[], double dt, double dx, double v, string output)
{
	double l_face[nx];
	double r_face[nx];
	double vdaj[a_len-2];
	double fl[nx+1];
	double fr[nx];
	double temp[a_len];

	// set index variables
	inter_start = ghost_num;
	inter_end = a_len-ghost_num;
	double mass;
	double fsum;
	double sum;
	int n=0;
	vector<double> l2_error;
	t = 0;
	ghost_num = 2; //SHOULD BE 2
	
	while (t<t_final)
	{
		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";
		
		// 	set ghost cells
		cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";


		// write solution to file
		cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		//-----------------------------------forward step----------------------------------//
		//---------------------------------------------------------------------------------//
		//--------------- reconstruction ------------------//
		cout << "Reconstructing profile\n";

		// need to initialize vdaj in main and pass in EMPTY VECTOR
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj2(a, a_len, i) ; //calculate daj for each point i=j
	
		for(int i=inter_start; i<=inter_end; i++)
		{
			l_face[i-ghost_num] =  a[i]-vdaj[i-ghost_num]/2.;
			r_face[i-ghost_num] =  a[i]+vdaj[i-ghost_num]/2.;		
		}
		//-------------------------------------------------//
			
		// ---------------flux calculation ----------------------//


		// calculate domain of dependence
		// ASSUMING CONSTANT DX
		// v is scalar and dt is static
		// only going to work for 2nd order: need better integration for 3rd order.......

		double y = v*dt;
		double al;
		double ar;
		double m;
		double b;
		double xl;
		double xr;
		double dx_trap;


		cout << "Calculating Flux\n";
		//*
		for(int i=0; i<nx+1; i++)
		{
			// -----------------------Left Flux---------------------//	
			// calculate slope and intercep to get line between j-1/2 and j-1/2 - y
			m = vdaj[i]; // vdaj for a[ghost_num]  is vdaj[0] 
			b = a[i+ghost_num-1]-x[i+ghost_num-1]*m;
			// set x values for right and left of integration boundary
			xl = x[i+ghost_num] - (dx/2.) - y ;
			xr = x[i+ghost_num] - (dx/2.);
			al = m*xl+b;
			ar = l_face[i];
			dx_trap = fabs(xl-xr);
		
			fl[i] = trap(dx_trap, al, ar)/y;
			// ------------------------------------------------------//	

			// -----------------------Right Flux---------------------//	
			// calculate slope and intercep to get line between j+1/2 and j+1/2 - y
			m = vdaj[i+1]; // vdaj for a[1] is vdaj[0]
			b = a[i+ghost_num]-x[i+ghost_num]*m;
			// set x values for right and left of integration boundary
			xl = x[i+ghost_num]+ (dx/2.) - y ;
			xr = x[i+ghost_num] + (dx/2.) ;
			al = m*xl+b;
			ar = r_face[i];
			dx_trap = fabs(xl-xr);

			if(i>0)
				fr[i-1]=fl[i];

			// ------------------------------------------------------//	
		}

		cout << "Update\n\n";
		fsum = 0;

		cout << "fl = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fl[i]<<" ";
		cout << " ]\n\n";

		cout << "fr = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fr[i]<<" ";
		cout << " ]\n\n";


		// save forward step to temporary variable
		for(int i=inter_start;i<=inter_end-1; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		l2_error.push_back(L2error(x, a, t, a_len, v));
		for(int i=ghost_num ; i <= a_len-ghost_num ; i++)
		{
			a[i] = temp[i];
		}

		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		// debug printout	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n";
		
		mass = totalMass(a, a_len);
		cout << "Total mass is: " << mass << "\n";
		
		n++;
		cout << "\n";
		cout<< "//----------------------------------------------------//\n";	

		//-------------------------------------------------//
		// Write L2 error
		writeError(l2_error, a_len, output, dt);

	}
	//end of while loop
}
//----------------------------------------------------------------------

//---------------------3rd order Solver--------------------------------------------------//
// with monotonizing and averaging
void solve3rdOrder(double a[], double x[], double dt, double dx, double v, string output)
{
	double l_face[nx];
	double r_face[nx];
	double vdaj[a_len-2];
	double fl[nx+1];
	double fr[nx];
	double temp[a_len];

	// set index variables
	inter_start = ghost_num;
	inter_end = a_len-ghost_num;
	double mass;
	double fsum;
	double sum;
	int n=0;
	vector<double> l2_error;
	t = 0;
	ghost_num = 2; //SHOULD BE 2
	
	while (t<t_final)
	{
		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";
		
		// 	set ghost cells
		cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";


		// write solution to file
		cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		//-----------------------------------forward step----------------------------------//
		//---------------------------------------------------------------------------------//
		//--------------- reconstruction ------------------//
		cout << "Reconstructing profile\n";

		// need to initialize vdaj in main and pass in EMPTY VECTOR
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj2(a, a_len, i) ; //calculate daj for each point i=j

		// Reconstruct using eq 1.9 from Coella	
		for(int i=inter_start; i<=inter_end; i++)
		{
			if(i<inter_end)
				r_face[i-ghost_num] =  (7./12.)*( a[i] + a[i+1] ) - (1./12.)*( a[i+2]+a[i-1]  );
			if(i>inter_start)	
				l_face[i] = r_face[i-1];
		}
		//-------------------------------------------------//
			
		// ---------------flux calculation ----------------------//

		// ASSUMING CONSTANT DX
		for(int i=0; i<nx+1; i++)
		{
			//double flux3rdOrderL(a, x, vdaj, l_face, r_face, a6j, i)
		}	
		cout << "Calculating Flux\n";
		/*
		for(int i=0; i<nx+1; i++)
		{
			// -----------------------Left Flux---------------------//	
			// calculate slope and intercep to get line between j-1/2 and j-1/2 - y
			m = vdaj[i]; // vdaj for a[ghost_num]  is vdaj[0] 
			b = a[i+ghost_num-1]-x[i+ghost_num-1]*m;
			// set x values for right and left of integration boundary
			xl = x[i+ghost_num] - (dx/2.) - y ;
			xr = x[i+ghost_num] - (dx/2.);
			al = m*xl+b;
			ar = l_face[i];
			dx_trap = fabs(xl-xr);
		
			fl[i] = trap(dx_trap, al, ar)/y;
			// ------------------------------------------------------//	

			// -----------------------Right Flux---------------------//	
			// calculate slope and intercep to get line between j+1/2 and j+1/2 - y
			m = vdaj[i+1]; // vdaj for a[1] is vdaj[0]
			b = a[i+ghost_num]-x[i+ghost_num]*m;
			// set x values for right and left of integration boundary
			xl = x[i+ghost_num]+ (dx/2.) - y ;
			xr = x[i+ghost_num] + (dx/2.) ;
			al = m*xl+b;
			ar = r_face[i];
			dx_trap = fabs(xl-xr);

			if(i>0)
				fr[i-1]=fl[i];

			// ------------------------------------------------------//	
		}
		*/

		cout << "Update\n\n";
		fsum = 0;

		cout << "fl = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fl[i]<<" ";
		cout << " ]\n\n";

		cout << "fr = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fr[i]<<" ";
		cout << " ]\n\n";


		// save forward step to temporary variable
		for(int i=inter_start;i<=inter_end-1; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		l2_error.push_back(L2error(x, a, t, a_len, v));
		for(int i=ghost_num ; i <= a_len-ghost_num ; i++)
		{
			a[i] = temp[i];
		}

		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		// debug printout	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n";
		
		mass = totalMass(a, a_len);
		cout << "Total mass is: " << mass << "\n";
		
		n++;
		cout << "\n";
		cout<< "//----------------------------------------------------//\n";	

		//-------------------------------------------------//
		// Write L2 error
		writeError(l2_error, a_len, output, dt);

	}
	//end of while loop
}
//----------------------------------------------------------------------



int main(int argc,char* argv[])
{
	if( remove( "scalar_advec_sol.out" ) != 0 )
    		perror( "Error deleting file" );
  	else
    		puts( "File successfully deleted" );

	
	// initialize simulations parameters
	int write_freq=1;
	vel = 1.0; // velocity
	nx=10; // number of nodes
	double c = .5;  //Courant number
	ghost_num = 2;
	int counter;
	// read in command line arguments. pass nx and c
    if(argc==1)
        cout << "\nNo Extra Command Line Argument, using default nx and c";
    if(argc>=2)
    {
		nx = atof(argv[1]);
		c = atof(argv[2]);
		cout << "given nx = " << atof(argv[1]) << " and c = " << atof(argv[2]) << "\n";
    }

	a_len = nx+(2*ghost_num); // defined length of solution/ghost node vector

	// initialize spatiall domain
	right_boundary = 2*M_PI;
	left_boundary  = 0.;
	double dx = (right_boundary-left_boundary)/(nx);

	// define dt to satisfy satisfy CFL condition
	double dt = c*(dx/fabs(vel));
	dt = dt; 
	dt = .001;

	t_final = 10;
	//t_final = 5*dt;
	t_start = 0.;


	cout << "\n\nComputing solution to scalar advection equations with "<<nx<<" cells\n\n" ;
	cout << "t_start is "<<t_start<<"\n" ;
	cout << "t_final is "<<t_final<<"\n" ;
	cout << "Velocity vel is "<< vel << "\n";
	cout << "nx is "<<nx<<"\n\n" ;
	cout << "dt is "<<dt<<"\n" ;
	cout << "dx is "<<dx<<"\n\n" ;


	// ----- initialize arrays -----//
	double a[a_len];
	double mem[a_len-1];
	double x[a_len];
	 
	 
	for(int i=0 ; i<=a_len-1 ; i++)
	{
		x[i] = left_boundary + i*dx;
		x[i] = x[i] - (((ghost_num - 1)+ 0.5) * dx) ;

	}
	
	//iterate through cells not boundary
	for(int i=0 ; i<=a_len-1 ; i++)
		a[i] = sin(x[i]);

	cout << "RJL2 " << a_len << "\n";

	//	Initialize Simulation Variables
	double mass;
	double temp[a_len]; //temp variable to hold a^n+1 during computation
	double xi[a_len]; //interpolation variable 
	double flux[a_len-1];
	double fluxR[a_len-2*ghost_num];
	double fluxL[a_len-2*ghost_num];

    int n = 0;
	double t = t_start;
	double fsum;
	vector<double> l2_error;
	string output = "l2_error";		

	//print out arrays for verification	
	/*
	cout << "Spatial domain: \n";
	for(int i=0 ; i<a_len ; i++)
	{
		cout << "x["<<i-1<<"] = "<<x[i]<<"\n";
		cout << "a["<<i-1<<",0] = "<<a[i]<<"\n\n";
	}
	for(int i=0 ; i<nstep+1 ; i++)
	{
		cout << "Time domain: \n";
		cout << "t = "<<t_start+i*dt<<"\n";
	}
	*/

	cout << "Initial Condition\n";	
	cout << "t = "<<t<<"\n";
	cout << "x = [ ";
		for(int i=0 ; i <= a_len-1 ; i++)
			cout << x[i]<<" ";
		cout << " ]\n\n";
	cout << "a = [ ";
		for(int i=0 ; i <= a_len-1; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";

	cout << "//--------------------------STARTING SIMULATION--------------------------------//\n";
	//solve1stOrder(a, x, dt,  dx, vel, output);
	solve2ndOrder(a, x, dt,  dx, vel, output);
	//solve3rdOrder(a, x, dt,  dx, vel, output);

	/*
	//------------Run simulation------------//
	// NEED TO CONDENSE FOR READABILITY!!!!!
	while( t<t_final )
	{
		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";
		
		// 	set ghost cells
		cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[i] = a[a_len-1-ghost_num-i];
				a[a_len-1-i] = a[ghost_num+i];
		}
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";


		// write solution to file
		cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		fsum= 0;
		//	compute membrane fluxes	
		for(int i=0 ; i < a_len-1 ; i++)
		{
			//flux[i] = membrane_flux(a[i],a[i+1],u);
			flux[i] = membrane_flux_linear(a[i],a[i+1],u);
			fsum= fsum + flux[i];
		}



		for(int i=0 ; i < a_len-2*ghost_num ; i++)
		{
			fluxL[i] = membrane_flux_linear(a[i+ghost_num],a[i+ghost_num-1],u);
			fluxR[i] = membrane_flux_linear(a[i+ghost_num],a[i+ghost_num+1],u);
		}

		// 	print membrane fluxes
		if(1)
		{	
				cout << "flux = [ ";
				for(int i=0 ; i < a_len-1 ; i++)
					cout << flux[i]<<" ";
				cout << " ]\n\n";
		}

		//	update all points
		cout << "Update\n\n";
		for(int i=ghost_num ; i < a_len ; i++)
		{
			temp[i] = a[i] - u * ( dt/dx ) * ( flux[i] - flux[i-1] );
		}	
		
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		l2_error.push_back(L2error(x, a, t, a_len, u));
		for(int i=ghost_num ; i < a_len-ghost_num ; i++)
		{
			a[i] = temp[i];
		}

		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		// debug printout	
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n";
		
		mass = totalMass(a, a_len);
		cout << "Total mass is: " << mass << "\n";
		
		n++;
		cout << "\n";
		cout<< "//----------------------------------------------------//\n";	
	}
	// Write L2 error
	writeError(l2_error, a_len, output, dt);
	*/
}

