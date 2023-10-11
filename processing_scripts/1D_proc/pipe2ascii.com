#!/bin/csh
#usage mul_drx2d.com firstnumber lastnumebr
if ($#argv > 2) then
        echo "usage: procall  firstnumber lastnumer"
        exit(1)
endif

if ($#argv > 0) then
        set i = ${argv[1]}
        if $#argv == 1 then
         set n = ${argv[1]}
        else
         set n = ${argv[2]}
        endif
        while ($i <= $n)
                echo "convert $i/test.ft1 to ascii"

pipe2txt.tcl -index PPM -real $i/test.ft1 > $i/test_1D.txt

@ i++

sleep 2
end
endif

