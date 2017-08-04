void eCheckIn (type typer , char *string)// int argc?
{
    // 0 type error, error string is input.
    if (!typer)
    {
        if (!rank) std::cout << string << std::endl;
        exit(-1);
    }
    if (argc < 6)
	{
        if (rank == 0)
        {
        std::cout << "NOT ENOUGH ARGUMENTS:" << std::endl;
		std::cout << "The Program takes 8 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << std::endl;
        std::cout << "Algorithm type, Variable Output File, Timing Output File (optional)" << std::endl;
        }
        exit(-1);
	}

	if ((dv & (tpb-1) != 0) || (tpb&31) != 0)
    {
        if (rank == 0)
        {
        std::cout << "INVALID NUMERIC INPUT!! "<< std::endl;
        std::cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << std::endl;
        }
        exit(-1);
    }


    if (dimz.dt_dx > .21)
    {
        if (rank == 0)
        {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.21 for stability." << endl;
        }
        exit(-1);
    }

}