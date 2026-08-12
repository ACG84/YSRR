# Drive-nonlinearity threshold vs carrier frequency

Permalloy vortex disk, 100 nm radius, Ms 800 kA/m, no bias. The measurement
that showed the element problem was a carrier-frequency problem rather than a
materials one.

`results_*_f{5,9,12}.json` are the SETTLED runs -- 40 drive cycles of settle,
2.0 ring-down time constants -- and are the numbers to quote. `sweep72_f*.json`
are the remaining points of the original 7.2-cycle sweep, kept because they
carry the mode structure (linear-response peaks at 5 and 9 GHz with an
anti-resonance at 7) even though they were read at 36% of tau.

    f (GHz)   threshold   ratio to alpha*omega/gamma
      5.00     1.50 mT      1.05     essentially on a mode
      9.00     6.00 mT      2.34
     12.00    16.00 mT      4.67     the operating point used all project

Reproduce with:

    python scripts/check_drive_nonlinearity.py --freq 9 --cycles 40 \
        --amps-mT 0.25 0.5 1 1.5 2 3 4 6 8 12 16 24 32 40 --device cuda
    python scripts/analyse_freq_threshold.py --dir data/freq_thresholds

Quote the compression curve rather than the threshold label where precision
matters: the criterion fires at norm 0.950 and the settled 1.5 mT row at 5 GHz
is 0.949, clearing its own cutoff by 0.001.
