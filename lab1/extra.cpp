
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
		writeSolution(a,x,t,mass,a_len,t_start);

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
		for(int i=0; i<nx; i++)
		{
			fr[i] = flux3rdOrderR(a, x,  l_face, r_face, v, i);
			if(i>nx)	
				fl[i] = fr[i-1];
		}	
		fl[0] = fr[nx-1];

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

		//l2_error.push_back(L2error(x, a, t, a_len, v));
		l2_error.push_back(L2error(x, a, t, dx, a_len, v));
		cout << "l2_error = [ ";
		for(int i=0 ; i < l2_error.size() ; i++)
			cout << l2_error[i]<<" ";
		cout << " ]\n";
		

		l1_error.push_back(L1error(x, a, t, dx, a_len, v));
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


