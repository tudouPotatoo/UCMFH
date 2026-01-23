#!/bin/bash

# Test script for UCMFH with HashNet weighted balance strategy

# Train with weighted balance strategy on mirflickr dataset (16 bits)
echo "Training with HashNet weighted balance strategy on mirflickr (16 bits)..."
python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance

# Train without weighted balance strategy for comparison
echo "Training without weighted balance strategy on mirflickr (16 bits)..."
python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1

# Test with weighted balance strategy on mirflickr dataset (32 bits)
echo "Training with HashNet weighted balance strategy on mirflickr (32 bits)..."
python demo.py --dataset mirflickr --hash_lens 32 --epoch 50 --task 1 --use_weighted_balance

# Test with weighted balance strategy on mscoco dataset
echo "Training with HashNet weighted balance strategy on mscoco (16 bits)..."
python demo.py --dataset mscoco --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance

# Test with weighted balance strategy on nus-wide dataset
echo "Training with HashNet weighted balance strategy on nus-wide (16 bits)..."
python demo.py --dataset nus-wide --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance
