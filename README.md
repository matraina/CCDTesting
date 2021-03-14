# CCDTesting Repo






HOW TO:

#####Directory structure#####
CCDTesting directory must contain directories CCDName/, which in turn will contain raw/ , processed/ , header/ and reports/ directories

#####Run main analysis script (usually automated to analyse and report on several images from e.g. parameter scan)#####
python3 main.py path/to/CCDTesting/directory raw/Image_Name processed/Img_Name starting_skip final_skip_avg_img final_skip_pcd_difference_img
