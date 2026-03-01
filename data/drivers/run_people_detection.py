#!/usr/bin/env python3
"""Example script to process the people-detection dataset."""

from people_detection import PeopleDetectionDriver

if __name__ == "__main__":
    dataset_path = "/Users/nadir/Documents/person-identification/data/datasets/people_detection"
    output_path = "/Users/nadir/Documents/person-identification/data/output"
    
    driver = PeopleDetectionDriver(dataset_path, output_path)
    driver.process()
    print("Dataset processing completed!")
