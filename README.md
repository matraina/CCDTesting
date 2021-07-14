# CCDTesting Repo

HOW TO:

#####Directory structure#####
Analysis directory (tipically CCDName/date/) must contain raw/, processed/ , header/ and reports/ directories if raw_processed_header_reports_dir_structure set to true in config.json
Alternatively, it is possible to have input and output files all in the same directory; this should be set in 'working_directory' in config.json 

#####Configuration file#####
- test should be stated in config file before running the corresponding analysis script. Current options: 'tweaking', 'linearity','quality','chargetransfer','clusters_depth_calibration'
- if run executable =/= from stated test code will ask to confirm


Structure of the configuration file:

{
    "test" : "tweaking",
    "working_directory" : "./", #mother directory where images or directory structure (raw/, processed/, etc.) are located
    "raw_processed_header_reports_dir_structure" : true, #using raw/, processed/, header, reports/ directory structure?
    "skip_start" : 5, #start skip used for single skip noise, difference and average image
    "skip_end" : -1, #end skip for difference and average image
    "fix_leach_reconstruction" : true, #correct reconstruction bug due to leach
    "reverse" : true, #number of electrons increasing (true) or decreasing (false) as ADU value decreases
    "ccd_register_size": 1036,  #size of register for overscan location (prescan+registersize+prescan+overscan)
    "analysis_region" : "full", #where to compute noise and dark current (anticlustering only). Accepted values: 'full', 'exposed_pixels', 'overscan', 'arbitrary'
    if analysis_region set to 'arbitrary', also change arguments below to meaningful/possible values to locate chosen region:
    "lower_row":-1, #if set to -1 fall back to full image
    "upper_row":-1,
    "lower_column":-1,
    "upper_column":-1,
    "kcl_threshold":3.2, #threshold in sigma units for charge loss coefficient PCDD tail counts (eg count below -3.2sigma and above 3.2sigma)
    "calibration_constant_guess" : 10, #guess for calibration constant used in fit and if calibration fails/not performed
    "print_header" : false, #print header.txt? (in /header or mother dir) 
    "tweaking_analysis":  #configure tweaking analysis
    [{
        "multiple_images" :
        [{
            "use_multiple_images" : false,
            "lower_index" : 2, #image_index at the end of image name. Eg Image_LPNHE_2
            "upper_index" : 3,
            "scan_report" : true, # report for a paramter scan 
            "scan_parameters": "VDD", #for which paramter
            "scan_intervals": "-17,-24,-2" #specify scan values (not used ftm)
        }],
        "report" : #compute and report on below objects (tweaking)
        [{
            "header" : true, #extract and print header in report
            "image" : true, #extract and print image sample region in report
            "pcds" : true,  #extract and print pcds in report
            "chargeloss" : true, #study and print chargeloss in report
            "calibration_darkcurrent" : true, #study and print calibration and dc estimations in report
            "fft_skips" : true, #study and print fft across skips in report
            "fft_row" : true #study and print fft across row pixels in report
        }]
    }],
    "linearity_analysis" : #extract information about ccd response linearity using a single (high exposure) image or multiple images (same or different params)
    [{
        "calibrate" : true, #perform calibration of average image pcd and use cal constant in linearity analysis 
        "max_electrons" : 2, #select the maximum n of electrons for comparison between measured and expected n electrons
        "multiple_images" : #configuration for multiple image usage
        [{
            "use_multiple_images" : true, #analysis using multiple images?
            "measured_vs_expected_e_with_multiple_images" : true, #cumulate imgs statistics electron by electron to check linearity (same-parameter imgs)
            "stddevs_vs_means_0_e_peaks" : true, #extract mean and sigma of 0-e peaks and assess linearity sigma=q+m*mean (different-parameter imgs)
            "lower_index" : 8, #lower index when multimg, in img dir we want imgs named: imagenameprefix+index (eg image8.fits,image9.fits,...,image20.fits)
            "upper_index" : 20 #upper index when multimg, in img dir we want imgs named: imagenameprefix+index (eg image8.fits,image9.fits,...,image20.fits)
        }],
        "report" : #compute and report on below objects (linearity)
        [{
            "header" : true, #extract and print header in report
            "image" : true, #extract and print image sample region in report
            "calibration_darkcurrent" : true, #print avg img pcd and gauss poisson fit (if chose to calibrate) (1st image pcd for multimg)
            "linearity_curves" : true #report available linearity curves plots
        }]
    }]
}


#####Run tweaking.py (linearity.py, etc.) analysis script (can be automated to analyze and report on several images from e.g. parameter scan)#####
./tweaking.py       raw_Image_Name        processed_Img_Name   #.fits must be omitted for both arguments    
e.g.: ./tweaking.py Image_LPNHE_16 Img_16


General Notes:
- start_skip for avg and diff starts at 0, not at 1, if 1 chosen, it will start from the second skip
- end_skip also follows the convention above, so to check the last skip of a 1000-skip image, it should be set to 999. This hassle can be dealt with as explained below:
- Setting -1 (or any other impossible value) for skip number arguments: start_skip and end_skip will set them to 0 (first skip) and N_tot_skip-1, respectively
- Set starting and end skips hold for all processing (avg image, std image, difference image (start+1-end to avoid using first skip), etc.) except for noise trend. The noise trend is computed for all skips regardless
- There is no .fits extension at the end of image name (for raw and for processed image) both in shell and python scripts

Acknowledgements:
- The fit framework was adapted based on Alex Piers example (check calibration module for further info)
- The code used few basic lines of plot_fits-image.py (By: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz. License: BSD) to display info and open .fits files with python
