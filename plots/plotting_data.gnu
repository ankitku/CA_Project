#!/usr/bin/gnuplot
#
reset

set terminal png 

unset key

set ytics 50
set tics scale 0.75

set xrange [1:31]
#set yrange [0:800]
set autoscale y

set xlabel 'iterations'
set ylabel 'timestamp difference'

set title 'timestamp difference when diverging recursively with atomic operation'
set output 'recursively_divergent_atomic.png'
plot 'recursively_divergent_clockdiff_atomic.dat' with linespoints pt 5

set title 'timestamp difference when diverging recursively with parallel operation'
set output 'recursively_divergent_parallel.png'
plot 'recursively_divergent_clockdiff_parallel.dat' with linespoints pt 5

set title 'timestamp difference when diverging iteratively with atomic operation'
set output 'iteratively_divergent_atomic.png'
plot 'iteratively_divergent_clockdiff_atomic.dat' with linespoints pt 5

set title 'timestamp difference when diverging iteratively with parallel operation'
set output 'iteratively_divergent_parallel.png'
plot 'iteratively_divergent_clockdiff_parallel.dat' with linespoints pt 5
