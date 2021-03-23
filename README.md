# CCDTesting Repo

HOW TO:

#####Directory structure#####
Analysis directory (tipically CCDName/date/ but can be adapted) must contain raw/, processed/ , header/ and reports/ directories
Using multi_Image_Analysis.sh, the latter 3 (processed/ header/ and reports/) will be created if not there, but having raw/ with .fits files is a requirement.


#####Shell analysis automation script#####
source multi_Image_analysis.sh      path/to/CCDTesting/directory    raw_Image_Name      processed_Image_Name      start_Index_In_Image_Name     end_Index_In_Image_Name        
e.g.: source multi_Image_Analysis.sh     /home/damic/data/Parameter_Scans/UW1603S/20210118   Image_LPNHE_VDD_    Img_VDD_   16     24
In the case of line above images: Image_LPNHE_VDD_16.fits, Image_LPNHE_VDD_17.fits, ..., Image_LPNHE_VDD_24.fits will be analyzed and processed into Img_VDD_16.fits, etc.
Headers and reports of all images will be in the respective directories inside path/to/CCDTesting/directory


#####Run main.py analysis script (usually automated to analyze and report on several images from e.g. parameter scan)#####
python3     main.py      path/to/CCDTesting/directory      raw/Image_Name        processed/Img_Name      starting_skip_avg_img_and_difference     final_skip_avg_img_and_difference    
e.g.: python3 main.py /home/damic/data/Parameter_Scans/UW1603S/20210118 raw/Image_LPNHE_VDD_16 processed/Img_VDD_16 5 -1


General Notes:
- starting_skip_avg_img_and_difference starts at 0, not at 1, if 1 chosen, it will start from the second skip
- final_skip_avg_img_and_difference follows the convention above, so to check the last skip of a 1000-skip image, it should be set to 999. This hassle can be dealt with as explained below:
- Setting -1 (or any other impossible value) for skip number arguments: starting_skip, final_skip_avg_img_and_difference will set them to 0 (first skip) and N_tot_skip-1, respectively
- When not starting with first skip, there will be a discrepancy in the noise trend plot in the report: the x-axis skip_number will effectively be the value shown minus starting_skip_avg_img_and_difference
- There is no .fits extension at the end of image name (for raw and for processed image) both in shell and python scripts
