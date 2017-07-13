#include <fstream>
#include <ostream>
#include <yaml.h>

// Something that will hold the output files and print the results.  Result print?  It could take strings of the details and the time/array.
class printer
{
    public:
    ofstream fwr;
    fwr.precision(10);

    void resultOut(REAL t, state *outstate)
    {   
        fwr << t << " ";
    }

    void resultFirst(REAL lx, REAL dv, REAL dx)
    {
        fwr << lx << " " << (dv-2) << " " << dx << " " << endl;
    }

    fwr << "Density " << t_eq << " ";
    for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << t_eq << " ";
    for (int k = 1; k<(dv-1); k++) fwr << T_f[k].y/T_f[k].x << " ";
    fwr << endl;

    fwr << "Energy " << t_eq << " ";
    for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
    fwr << endl;

    fwr << "Pressure " << t_eq << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
    fwr << endl;
}
