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
//assumes constant dx
double dx;
double dt;

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
double L2error(double x[], double a[], double t, double dx, int len_a, double u)
{
	double L2_error=0;
	double analytical = 0;
	double difference = 0;
	for(int i=ghost_num;i<len_a-ghost_num;i++)
	{
		analytical = sin(x[i] - u * t);
		difference = analytical - a[i];
		L2_error += pow(difference * dx, 2.);
		// cout << pow(a[i] - sin(x[i]-u*t), 2.) << "\n";
	}
	L2_error = sqrt(L2_error);
	return L2_error;
}
//----------------------------------------------------------------------
double L1error(double x[], double a[], double t, double dx, int len_a, double u)
{
	double L1_error=0;
	double analytical = 0;
	double difference = 0;
	for(int i=ghost_num;i<len_a-ghost_num;i++){
		analytical = sin(x[i] - u * t);
		difference = analytical - a[i];
		L1_error += fabs(difference * dx);
	}
	// L1_error /= nx;
	return L1_error;
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
	{
		daj_m = smallest( fabs(daj) , 2.*fabs(a[j]-a[j-1]), 2.*fabs(a[j]-a[j+1]) );
		daj_m *= (daj/fabs(daj));
	}
	else
	{
		daj_m = 0;
	}
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
	
	cout << "\ndaj before: " << daj << "\n";

	//--- monotonization ---//
	//cout << "\ndaj before: " << daj << "\n";
	if(1)
	{
		if( ((a[j+1] - a[j]) * (a[j] - a[j-1])) > 0.)
		{
			daj_m = smallest( fabs(daj) , 2.*fabs(a[j]-a[j-1]), 2.*fabs(a[j+1]-a[j])) ;
			//cout << "daj_m before: " << daj_m << "\n";
			daj_m *= (daj/fabs(daj));
			//cout << "daj_m after: " << daj_m << "\n";
		}	
		else
		{
			daj_m = 0;
		}
		return daj_m;
	}
	else
		return daj;
	//----------------------//

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
double flux3rdOrderR(double a[],double x[],  double l_face[], double r_face[],double v, int j)
{
	// Calculated 3rd order flux at right membrane of jth node
	// Only works for postive velocity!
	if(vel<0)
	{
		cout << "need to use upwinding from the right direction!!";
		exit(0);
	}

	double y = v*dt;
	double arj=r_face[j];
	double alj=l_face[j];
	double a6_j = a6j( alj, arj, a[j+ghost_num]);
	double x_coella = y/dx;
	double val;
	val =  arj - (x_coella/2.) * ( arj - alj * (1 - (2./3.)*x_coella)*a6_j )  ;
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

	// cout << "Sum of TEMP vector is: " << sum << "\n";
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
	for(int i = ghost_num ; i < a_len-ghost_num ; i++)	
		mass += x[i] * dx;
	// cout << dx << endl;
	return mass;
}



//---------------------1st order Solver--------------------------------------------------//
// with monotonizing and averaging
void solve1stOrder(double a[], double x[], double dt, double dx, double v, string output1, string output2)
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
	vector<double> l1_error;
	t = 0;
	ghost_num = 2; //SHOULD BE 2
	
	while (t<t_final)
	{
		//	continue iterating through time while step n is less than step total
	
		// cout<< "//------------t = "<< t <<"------------//\n";	
		// cout << "a = [ ";
		// for(int i=0 ; i < a_len ; i++)
		// 	cout << a[i]<<" ";
		// cout << " ]\n\n";
		
		// 	set ghost cells
		// cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				// cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				// cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		// cout << "a = [ ";
		// for(int i=0 ; i < a_len ; i++)
		// 	cout << a[i]<<" ";
		// cout << " ]\n\n";


		// write solution to file
		// cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		mass = totalMass(a, a_len);
		cout << "Total mass is: " << mass << "\n";

		//-----------------------------------forward step----------------------------------//
		//---------------------------------------------------------------------------------//


		//--------------- reconstruction ------------------//
		// cout << "Reconstructing profile\n";
		// reconstruction using <double> vector s
		//recon(a, x, a_len, dt, v, l_face, r_face, vdaj);

		/*
		// need to initialize vdaj in main and pass in EMPTY VECTOR
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj1(a, a_len, i) ; //calculate daj for each point i=j
	
		for(int i=inter_start; i<=inter_end; i++)
		{
			l_face[i-ghost_num] =  a[i]-vdaj[i-ghost_num]/2.;
			r_face[i-ghost_num] =  a[i]+vdaj[i-ghost_num]/2.;		
		}
		*/
		//-------------------------------------------------//
			
		// --------------- 1st order flux calculation ----------------------//
		// cout << "Calculating Flux\n";
		for(int i=0; i<nx; i++)
			fl[i] = a[i+ghost_num-1];

		for(int i=0; i<nx; i++)
			fr[i] = a[i+ghost_num];
		// ------------------------------------------------------//	


		// // Print Left and Right Fluxes for all nodes
		// cout << "fl = [ ";
		// for(int i=0 ; i < nx ; i++)
		// 	cout << fl[i]<<" ";
		// cout << " ]\n\n";

		// cout << "fr = [ ";
		// for(int i=0 ; i < nx ; i++)
		// 	cout << fr[i]<<" ";
		// cout << " ]\n\n";


		// cout << "Update\n\n";
		// save forward step to temporary variable
		fsum = 0;
		for(int i=inter_start;i<=inter_end-1; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}

		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		// cout << "After update we have\n";
		// cout << "fsum = "<<fsum<<"\n";
		for(int i=inter_start ; i <= inter_end-1 ; i++)
		{
			a[i] = temp[i];
		}

		l2_error.push_back(L2error(x, a, t, dx, a_len, v));
		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		l1_error.push_back(L1error(x, a, t, dx, a_len, v));
		cout << "l1_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l1_error[i]<<" ";
		cout << " ]\n";
		

		// debug printout	

		cout << "xr = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << x[i]<<" ";
		cout << " ]\n";
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
	//end of while loop
		//-------------------------------------------------//
		// Write L2 error
		writeError(l2_error, a_len, output1, dt);
		
}
//----------------------------------------------------------------------

//---------------------2nd order solveSolver--------------------------------------------------//
// with monotonizing and averaging
void solve2ndOrder(double a[], double x[], double dt, double dx, double v, string output1, string output2)
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
	vector<double> l1_error;
	t = 0;
	ghost_num = 2; //SHOULD BE 2
	
	while (t<t_final)
	{
		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		// cout << "a = [ ";
		// for(int i=0 ; i < a_len ; i++)
		// 	cout << a[i]<<" ";
		// cout << " ]\n\n";
		
		// 	set ghost cells
		// cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				// cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				// cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		// cout << "a = [ ";
		// for(int i=0 ; i < a_len ; i++)
		// 	cout << a[i]<<" ";
		// cout << " ]\n\n";


		// write solution to file
		cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		//------------------------------2nd Order forward step-----------------------------//
		//---------------------------------------------------------------------------------//
		//--------------- reconstruction ------------------//
		cout << "Reconstructing profile\n";

		// need to initialize vdaj in main and pass in EMPTY VECTOR
		// vdaj OFFSET BY 1 FROM a
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj1(a, a_len, i) ; //calculate daj for each point i=j
	
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


		// cout << "Calculating Flux\n";
		//*
		for(int i=0; i<nx+1; i++)
		{
			//ITERATING THROUGH NX+1 WILL CREATE EXTRA FL VALUE BUT THAT DONT MATTER
			// -----------------------Left Flux---------------------//	
			// calculate slope and intercep to get line between j-1/2 and j-1/2 - y
			m = vdaj[i]; // vdaj for a[1]  is vdaj[0] 
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
			{
				fr[i-1]=fl[i];
			}

			// ------------------------------------------------------//	
		}

		// cout << "Update\n\n";
		fsum = 0;

		// cout << "fl = [ ";
		// for(int i=0 ; i < nx ; i++)
		// 	cout << fl[i]<<" ";
		// cout << " ]\n\n";

		// cout << "fr = [ ";
		// for(int i=0 ; i < nx ; i++)
		// 	cout << fr[i]<<" ";
		// cout << " ]\n\n";


		// save forward step to temporary variable
		for(int i=inter_start;i<inter_end; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		for(int i=inter_start;i<inter_end; i++)
		{
			a[i] = temp[i];
		}

		l2_error.push_back(L2error(x, a, t, dx, a_len, v));
		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		
		l1_error.push_back(L1error(x, a, t, dx, a_len, v));
		cout << "l1_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l1_error[i]<<" ";
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
		}
	//end of while loop
	writeError(l2_error, a_len, output1, dt);
	writeError(l1_error, a_len, output2, dt);


}
//----------------------------------------------------------------------

//---------------------3rd order solveSolver--------------------------------------------------//
// modeled after 2nd
// with monotonizing and averaging
void solve3rdOrder_new(double a[], double x[], double dt, double dx, double v, string output1, string output2)
{
	double l_face[nx];
	double r_face[nx];
	double vdaj[a_len-2];
	double fl[nx];
	double fr[nx+1];
	double temp[a_len];

	// set index variables
	inter_start = ghost_num;
	inter_end = a_len-ghost_num;
	double mass;
	double fsum;
	double sum;
	int n=0;
	vector<double> l2_error;
	vector<double> l1_error;
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

		//------------------------------3rd Order forward step-----------------------------//
		//---------------------------------------NEW---------------------------------------//
		//--------------- reconstruction ------------------//
		cout << "Reconstructing profile\n";

		// need to initialize vdaj in main and pass in EMPTY VECTOR
		// vdaj OFFSET BY 1 FROM a
		for(int i=1; i<=a_len-1; i++)
			vdaj[i-1] = daj2(a, a_len, i) ; //calculate daj for each point i=j
	
		cout << "vdaj = [ ";
		for(int i=0 ; i < a_len-2 ; i++)
			cout << vdaj[i]<<" ";
		cout << " ]\n\n";


		// Reconstruct using eq 1.9 from Coella	
		// assumes equal spacing in x
		cout << "inter_start = " << inter_start << "\n";
		cout << "inter_end = " << inter_end << "\n";


		for(int i=inter_start; i<=inter_end; i++)
		{
			if(0)
			{
				if(i<inter_end)
					r_face[i-ghost_num] =  (7./12.)*( a[i] + a[i+1] ) - (1./12.)*( a[i+2]+a[i-1]  );
				if(i>inter_start)	
					l_face[i-ghost_num] = r_face[i-ghost_num-1];
			}
			else
			{
				if(i<inter_end)
					r_face[i-ghost_num] = a[i] + (1./2.)*( a[i+1] - a[i]  ) + (1./4.)*((2./3.)*vdaj[i-1] + (2./3.)*vdaj[i-1]); // NOTE: daj for a[i+1] is vdaj[i-1]
				if(i>inter_start)	
					l_face[i-ghost_num] = r_face[i-ghost_num-1];
			}
		}
		l_face[0] = r_face[nx-1];

		cout << "l_face = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << l_face[i]<<" ";
		cout << " ]\n\n";

		cout << "r_face = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << r_face[i]<<" ";
		cout << " ]\n\n";

		//-------------------------------------------------//
		


		// calculate domain of dependence
		// ASSUMING CONSTANT DX
		// v is scalar and dt is static
		// only going to work for 2nd order: need better integration for 3rd order.......
		// ------------------------------flux calculation ----------------------//
		// ASSUMING CONSTANT DX
		for(int i=0; i<nx+1; i++)
		{
			fr[i] = flux3rdOrderR(a, x,  l_face, r_face, v, i);
			if(i<nx)	
				fl[i] = fr[i-1];
		}	
		fl[0] = fr[nx];

		cout << "Calculating Flux\n";
	
		fsum = 0;
		cout << "fl = [ ";
		for(int i=0 ; i < nx ; i++)
		{
			cout << fl[i]<<" ";
			fsum += fl[i];
		}
		cout << " ]\n\n";

		cout << "fr = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fr[i]<<" ";
		cout << " ]\n\n";
		// ---------------------------------------------------------------------//
		cout << "Update\n\n";
		// save forward step to temporary variable
		for(int i=inter_start;i<inter_end; i++)
		{
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);
			fsum =+ (fl[i-ghost_num] - fr[i-ghost_num]);

		}


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		cout << "fsum = "<<fsum<<"\n";
		for(int i=inter_start;i<inter_end; i++)
		{
			a[i] = temp[i];
		}

		l2_error.push_back(L2error(x, a, t, dx, a_len, v));
		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		l1_error.push_back(L2error(x, a, t, dx, a_len, v));
		cout << "l1_error = [ ";
		for(int i=0 ; i < l1_error.size() ; i++)
			cout << l1_error[i]<<" ";
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
		writeError(l2_error, a_len, output1, dt);
		writeError(l1_error, a_len, output2, dt);

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
		//	continue iterating through time while step n is less than step total
	
		cout<< "//------------t = "<< t <<"------------//\n";	
		// cout << "a = [ ";
		// for(int i=0 ; i < a_len ; i++)
		// 	cout << a[i]<<" ";
		// cout << " ]\n\n";
		
		// 	set ghost cells
		// cout << "Ghost update:\n";
		for(int i=0; i<ghost_num;i++)
		{
				// cout << "Setting a[" << i << "] to a["<<a_len-1-ghost_num-i <<"]\n";
				// cout << "Setting a[" << a_len-1-i<< "] to a["<<ghost_num+i <<"]\n\n";
				a[ghost_num-1-i] = a[a_len-1-ghost_num-i];
				a[a_len-ghost_num+i] = a[ghost_num+i];
		}
		cout << "a = [ ";
		for(int i=0 ; i < a_len ; i++)
			cout << a[i]<<" ";
		cout << " ]\n\n";


		// write solution to file
		// cout << "Writing solution to file\n\n";
		writeSolution(a,t,mass,a_len,t_start);

		//-----------------------------3rd order forward step------------------------------//
		//---------------------------------------------------------------------------------//
		

		//------------------ reconstruction ---------------------//
		// cout << "Reconstructing profile\n";
		// need to initialize vdaj in main and pass in EMPTY VECTOR
		//JUST MADE for(int i=1; i<=a_len-1; i++)
		for(int i=1; i<a_len-1; i++)
			vdaj[i-1] = daj2(a, a_len, i) ; //calculate daj for each point i=j

		// cout << "vdaj = [ ";
		// for(int i=0 ; i < a_len-2 ; i++)
		// 	cout << vdaj[i]<<" ";
		// cout << " ]\n\n";


		// Reconstruct using eq 1.9 from Coella	
		// assumes equal spacing in x
		// no monotonization!!
		// cout << "inter_start = " << inter_start << "\n";
		// cout << "inter_end = " << inter_end << "\n";
		for(int i=inter_start; i<=inter_end; i++)
		{
			// if(0)
			// {
			
			r_face[i-ghost_num] =  (7./12.)*( a[i] + a[i+1] ) - (1./12.)*( a[i+2]+a[i-1]  );


			// }
			// else
			// {
			// 	if(i<inter_end)
			// 		r_face[i-ghost_num] = a[i] + (1./2.)*( a[i+1] - a[i]  ) + (1./4.)*((2./3.)*vdaj[i-1] + (2./3.)*vdaj[i-1]); // NOTE: daj for a[i+1] is vdaj[i-1]
			// 	if(i>inter_start)	
			// 		l_face[i-ghost_num] = r_face[i-ghost_num-1];
			// }
		}
		for (int i = inter_start; i < inter_end; i++) {
			if (i-ghost_num == 0) {
				l_face[i-ghost_num] = r_face[inter_end-ghost_num];
			}
			else {
				l_face[i-ghost_num] = r_face[i-ghost_num-1];
			}
		}

		cout << "l_face = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << l_face[i]<<" ";
		cout << " ]\n\n";

		cout << "r_face = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << r_face[i]<<" ";
		cout << " ]\n\n";

		for (int i = 0; i < nx; i++) {
			fr[i] = r_face[i];
			fl[i] = r_face[i];
		}
		//-------------------------------------------------------//
	

		
		// // ------------------------------flux calculation ----------------------//
		// // ASSUMING CONSTANT DX
		// if(0)
		// {
		// 	for(int i=0; i<=nx+1; i++)
		// 	{
		// 		if(i<nx)
		// 			fr[i] = flux3rdOrderR(a, x,  l_face, r_face, v, i);
		// 		if(i>0)	
		// 			fl[i] = fr[i-1];
		// 	}	
		// 	fl[0] = fr[nx-1];
		// }
		// if(1)	// FROM
		// {		// solve3rdOrder_NEW
			
		// 	for(int i=0; i<nx+1; i++)
		// 	{
		// 		fr[i] = flux3rdOrderR(a, x,  l_face, r_face, v, i);
		// 		if(i<nx)	
		// 			fl[i] = fr[i-1];
		// 	}	
		// 	fl[0] = fr[nx];
		// } 



		cout << "Calculating Flux\n";
	
		fsum = 0;
		cout << "fl = [ ";
		for(int i=0 ; i < nx ; i++)
		{
			cout << fl[i]<<" ";
			fsum += fl[i];
		}
		cout << " ]\n\n";

		cout << "fr = [ ";
		for(int i=0 ; i < nx ; i++)
			cout << fr[i]<<" ";
		cout << " ]\n\n";
		// ---------------------------------------------------------------------//



		cout << "fsum = " << fsum << "\n";
		cout << "Update\n\n";
		// save forward step to temporary variable
		for(int i=inter_start;i<=inter_end-1; i++)
			temp[i] = advect(a, dt,  dx,  v, fl, fr, i);


		// NEED TO CONDENSE FOR READABILITY!!!!!
		t += dt;
		cout << "After update we have\n";
		for(int i=inter_start ; i <= inter_end ; i++)
		{
			a[i] = temp[i];
		}

		l2_error.push_back(L2error(x, a, t, dx, a_len, v));
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
		cout << "given nx = " << nx << " and c = " << c << "\n";
    }

	a_len = nx+(2*ghost_num); // defined length of solution/ghost node vector

	// initialize spatiall domain
	right_boundary = 2*M_PI;
	left_boundary  = 0.;
	dx = (right_boundary-left_boundary)/(nx);

	// define dt to satisfy satisfy CFL condition
	dt = c*(dx/fabs(vel));
	dt = dt; 

	t_final = 1;
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

	//cout << "RJL2 " << a_len << "\n";

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
	string output1 = "l2_error";		
	string output2 = "l1_error";		

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
	// solve1stOrder(a, x, dt,  dx, vel, output1, output2);
	// solve2ndOrder(a, x, dt,  dx, vel, output1, output2);
	//solve3rdOrder_new(a, x, dt,  dx, vel, output1, output2);
	solve3rdOrder(a, x, dt,  dx, vel, output1);

}
