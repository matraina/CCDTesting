# CCDTesting Repo

HOW TO:

#####Directory structure#####
Analysis directory (tipically CCDName/date/) must contain raw/, processed/ , header/ and reports/ directories if raw_processed_header_reports_dir_structure set to true in config.json
Using multi_Image_Analysis.sh, the latter 3 (processed/ header/ and reports/) will be created if not there, but having raw/ with .fits files is a requirement (if raw_processed_header_reports_dir_structure set to true in config.json).


#####Shell analysis automation script#####
source multi_Image_analysis.sh      path/to/CCDTesting/directory    raw_Image_Name      processed_Image_Name      start_Index_In_Image_Name     end_Index_In_Image_Name        
e.g.: source multi_Image_Analysis.sh     /home/damic/data/Parameter_Scans/UW1603S/20210118   Image_LPNHE_VDD_    Img_VDD_   16     24
In the case of line above images: Image_LPNHE_VDD_16.fits, Image_LPNHE_VDD_17.fits, ..., Image_LPNHE_VDD_24.fits will be analyzed and processed into Img_VDD_16.fits, etc.
Headers and reports of all images will be in the respective directories inside path/to/CCDTesting/directory (if selected in config.json)

#####Configuration file#####
Structure of the configuration file:
{
    "working_directory" : "/Users/mtraina/Downloads/raw1/", #mother directory where images or directory structure (raw/, processed/, etc.) are located
    "raw_processed_header_reports_dir_structure" : false, #using raw/, processed/, header, reports/ directory structure?
    "skip_start" : 5, #start skip used for single skip noise, difference and average image 
    "skip_end" : -1, #end skip for difference and average image
    "fix_leach_reconstruction" : true, #correct reconstruction bug due to leach
    "reverse" : true, #number of electrons increasing (true) or decreasing (false) as ADU value decreases
    "ccd_register_size": 1036, #size of register for overscan location
    "analysis_region" : "overscan", #where to compute noise and dark current (anticlustering only)
    "print_header" : false, #print header.txt in header or mother dir
    "report" : #compute and report on below objects
    [{
        "header" : true, #print header in report
        "image" : true, #print start,end,avg,std,diff01skips,diffStartEnd selected regions with cluster for rapid visual assessment
        "pcds" : true, #start and avg imgs pcds with fit and noise
        "chargeloss" : true, #diffStartEnd PCDD and compute kcl and skewness
        "calibration_darkcurrent" : true, #calibrate and compute dc with gausspoisson fit
        "fft_skips" : true, #perform fft across skip time sequence and average over all pixel
        "fft_row" : true #perform fft across row time sequence and average over row pixels
    }]
}

#####Run main.py analysis script (usually automated to analyze and report on several images from e.g. parameter scan)#####
python3      main.py       raw_Image_Name        processed_Img_Name    
e.g.: python3 main.py Image_LPNHE_VDD_16 Img_VDD_16


General Notes:
- start_skip for avg and diff starts at 0, not at 1, if 1 chosen, it will start from the second skip
- end_skip also follows the convention above, so to check the last skip of a 1000-skip image, it should be set to 999. This hassle can be dealt with as explained below:
- Setting -1 (or any other impossible value) for skip number arguments: start_skip and end_skip will set them to 0 (first skip) and N_tot_skip-1, respectively
- Set starting and end skips hold for all processing (avg image, std image, difference image (start+1-end to avoid using first skip), etc.) except for noise trend. The noise trend is computed for all skips regardless
- There is no .fits extension at the end of image name (for raw and for processed image) both in shell and python scripts
