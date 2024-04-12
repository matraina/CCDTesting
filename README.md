# CCDTesting Repo

HOW TO:

#####Directory structure#####
Working directory must be specified in 'working_directory' in config.json. If 'raw_processed_header_reports_dir_structure' is set to true, working directory must contain raw/ , processed/ , and reports/ directories if raw_processed_header_reports_dir_structure set to true in config.json

#####Configuration file#####
- the type of analysis should be stated in 'test' in config.json before running the corresponding analysis script. Current options: 'tweaking', 'linearity','quality','chargetransfer','clusters_depth_calibration'
- if run executable is different from stated test code will ask to confirm

#####Run tweaking.py (linearity.py, etc.) analysis script (can be automated to analyze and report on several images from e.g. parameter scan)#####
./tweaking.py       raw_Image_Name        processed_Img_Name   #.fits must be omitted for both arguments    
e.g.: ./tweaking.py Image_LPNHE_16 Img_16

General Notes:
- start_skip for avg and diff starts at 0. If 1 is input, it will start from the second skip
- end_skip also follows the convention above, so for a 1000-skip image it should be set to 999. 
- Setting am impossible value for start/end skip numbers will set start_skip and end_skip to 0 (first skip) and N_tot_skip-1, respectively
- Set start and end skips hold for all processing (avg image, std image, difference image (start+1-end to avoid using first skip), etc.) except for noise trend. The noise trend is computed for all skips regardless

Acknowledgements:
- The Gauss-Poisson convolution fit was adapted from Alex Piers' damicm-image-preproc (https://github.com/alexanderpiers/damicm-image-preproc)
- The code uses few basic lines of plot_fits-image.py (By: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz. License: BSD) to display info and open .fits files with python
