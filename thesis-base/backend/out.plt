set xlabel "Time"
set ylabel "Number of Chemical Species"
p "out.dat" u 1:2 t "X0" w l,"out.dat" u 1:3 t "X1" w l,"out.dat" u 1:4 t "X2" w l
set term png
set out "out.png"
rep
pause -1 'Enter'
