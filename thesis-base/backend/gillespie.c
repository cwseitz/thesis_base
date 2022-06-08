
/*
 20100106 masa
 A Sample of Gillespie Algorithm (Direct Method) for Autocatalytic Reaction Cycle with C
 
 Refer to:
 EGillespie, D.T., Exact stochastic simulation of coupled chemical reactions,
 The journal of physical chemistry, 81(25), 2340--2361, 1977
 http://pubs.acs.org/doi/abs/10.1021/j100540a008 


 How to compile:
 > gcc gillespie.c -lm

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
/////////////////////////////////////////////////////////////
//
// user definition part
//
#define ENDTIME   1.0		// end of time
#define TIMESTEP  0.005		// interval of output
#define OUTFILE	  "out.dat"	// output file
#define PLTFILE   "out.plt"	// gnuplot script file 
#define PNGFILE   "out.png"	// png file 
#define SEED      123		// random seed
#define N         6		// number of reaction
#define M         3		// number of chemical species

int x[M];			// population of chemical species
double c[N];			// reaction rates
double p[N];			// propencities of reaction
int s[N][M];			// data structure for updating x[]

	
void init(int x[], double c[], int s[][M]){

	// population of chemical species
	x[0] = 0;
	x[1] = 0;
	x[2] = 0;

	// reaction rates
	c[0] = 50;
	c[1] = 50;
	c[2] = 50;
	c[3] = 50;
	c[4] = 50;
	c[5] = 50;

	//gene activation
	s[0][0] = 1;
	s[0][1] = 0;
	s[0][2] = 0; 

	//gene deactivation
	s[1][0] = -1;
	s[1][1] = 0;
	s[1][2] = 0;
	
	//transcription
	s[2][0] = 0;
	s[2][1] = 1;
	s[2][2] = 0; 

	//RNA degradation
	s[3][0] = 0;
	s[3][1] = -1;
	s[3][2] = 0; 

	//translation
	s[4][0] = 0;
	s[4][1] = 0;
	s[4][2] = 1; 

	//protein degradation
	s[5][0] = 0;
	s[5][1] = 0;
	s[5][2] = -1; 

}
// function for updating the reaction propencities
void update_p(double p[], double c[], int x[]){
	p[0] = c[0]*(1-x[0]);
	p[1] = c[1]*x[0];
	p[2] = c[2]*x[0];
	p[3] = c[3]*x[1];
	p[4] = c[4]*x[1];
	p[5] = c[5]*x[2];
	for (int i=0; i<N; i++){
	    printf("%f\n",p[i]);
	}
}



int select_reaction(double p[], int pn, double sum_propencity, double r){
	int reaction = -1;
	double sp = 0.0;
	int i;
	r = r * sum_propencity;
	for(i=0; i<pn; i++){
		sp += p[i];
		if(r < sp){
			reaction = i;
			break;
		}
	}
	return reaction;
}

void update_x(int x[], int s[][M], int reaction){
	int i;
	for(i=0; i<M; i++){
		x[i] += s[reaction][i];
		if (x[i] < 0){
		    x[i] = 0;
		}
	}
}

void output(FILE *out, double t, int x[], int xn){
	static double output_t = 0.0;
	int i;
	if(output_t <= t){
		fprintf(out, "%f", t);
		for(i=0; i<xn; i++){
			fprintf(out, "\t%d", x[i]); 
		}
		fprintf(out, "\n");
		output_t += TIMESTEP;
	}
}


void output_gnuplot(int n){
	int i;
	FILE *out = fopen(PLTFILE,"w");
	char *outfile = OUTFILE;
	char *pngfile = PNGFILE;
	
	fprintf(out, "set xlabel \"Time\"\n");
	fprintf(out, "set ylabel \"Number of Chemical Species\"\n");
	fprintf(out, "p ");
	for(i=0; i<n; i++){
		if(i>0) fprintf(out, ",");
		fprintf(out, "\"%s\" u 1:%d t \"X%d\" w l", outfile, i+2, i);
	}
	fprintf(out, "\n");
	fprintf(out, "set term png\n");
	fprintf(out, "set out \"%s\"\n", pngfile);
	fprintf(out, "rep\n");
	fprintf(out, "pause -1 'Enter'\n");

	printf("Type the following command if gnuplot is installed in your computer.\n");
	printf(">gnuplot %s\n",pngfile);

	fclose(out);

}

double sum(double a[], int n){
	int i;
	double s=0.0;
	for(i=0; i<n; i++) 
		s += a[i];
	return(s);
}

int main(void){

	// initialization
	double sum_propencity = 0.0;	// sum of propencities
	double tau=0.0;			// step of time
	double t=0.0;			// time
	double r;			// random number
	int reaction;			// reaction number selected

	init(x, c, s);

	srand(SEED);
	
	FILE *out=fopen(OUTFILE, "w");
	
	// main loop
	while(t < ENDTIME){
	
		printf("Time step: %f, Counts: %d,%d,%d\n",t,x[0],x[1],x[2]);
		// output
		output(out, t, x, M);
	
		// update propencity
		update_p(p, c, x);
		sum_propencity = sum(p, N);
	

		// sample tau
		if(sum_propencity > 0){
			tau = -log((double)rand()/INT_MAX) / sum_propencity;
		}else{
			break;
		}
	
		// select reaction
		r = (double)rand()/INT_MAX;
		reaction = select_reaction(p, N, sum_propencity, r);
		printf("Reaction: %d\n,%f", reaction,tau);
	
		// update chemical species
		update_x(x, s, reaction);	
	
		// time
		t += tau;
	
	}
	fclose(out);
	
	// for gnuplot script
	output_gnuplot(M);

	return(0);
}

