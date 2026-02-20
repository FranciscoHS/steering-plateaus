#!/usr/bin/env python3
"""
Quick test: do Qwen3-1.7B and 4B reliably complete Java, C, Rust, Go, Lisp?
"""

import sys, argparse
sys.path.insert(0, '.')

import torch
from src.model import load_model

PROMPTS = {
    'java': [
        "public static int fibonacci(int n) {",
        "public static boolean isPrime(int n) {",
        "public static String reverseString(String s) {",
        "public static int[] sortList(int[] arr) {",
        "public static int binarySearch(int[] arr, int target) {",
        "public static double calculateDistance(double x1, double y1, double x2, double y2) {",
        "public static int gcd(int a, int b) {",
        "public static int maxValue(int[] arr) {",
        "public static boolean validateEmail(String email) {",
        "public static List<Integer> flatten(List<List<Integer>> nested) {",
    ],
    'c': [
        "int fibonacci(int n) {",
        "int is_prime(int n) {",
        "char* reverse_string(const char* s) {",
        "void sort_list(int* arr, int n) {",
        "int binary_search(int* arr, int n, int target) {",
        "double calculate_distance(double x1, double y1, double x2, double y2) {",
        "int gcd(int a, int b) {",
        "int max_value(int* arr, int n) {",
        "int validate_email(const char* email) {",
        "void matrix_multiply(int** a, int** b, int** result, int n) {",
    ],
    'rust': [
        "fn fibonacci(n: u32) -> u32 {",
        "fn is_prime(n: u32) -> bool {",
        "fn reverse_string(s: &str) -> String {",
        "fn sort_list(arr: &mut Vec<i32>) {",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {",
        "fn calculate_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {",
        "fn gcd(a: u32, b: u32) -> u32 {",
        "fn max_value(arr: &[i32]) -> i32 {",
        "fn validate_email(email: &str) -> bool {",
        "fn matrix_multiply(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {",
    ],
    'go': [
        "func fibonacci(n int) int {",
        "func isPrime(n int) bool {",
        "func reverseString(s string) string {",
        "func sortList(arr []int) []int {",
        "func binarySearch(arr []int, target int) int {",
        "func calculateDistance(x1, y1, x2, y2 float64) float64 {",
        "func gcd(a, b int) int {",
        "func maxValue(arr []int) int {",
        "func validateEmail(email string) bool {",
        "func matrixMultiply(a, b [][]int) [][]int {",
    ],
    'lisp': [
        "(defun fibonacci (n)",
        "(defun is-prime (n)",
        "(defun reverse-string (s)",
        "(defun sort-list (lst)",
        "(defun binary-search (lst target)",
        "(defun calculate-distance (x1 y1 x2 y2)",
        "(defun gcd-of (a b)",
        "(defun max-value (lst)",
        "(defun validate-email (email)",
        "(defun matrix-multiply (a b)",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen3-1.7b')
parser.add_argument('--max_tokens', type=int, default=60)
args = parser.parse_args()

print(f"Loading model: {args.model}")
model, tokenizer, device = load_model(args.model)

for lang, prompts in PROMPTS.items():
    print(f"\n{'='*70}")
    print(f"  {lang.upper()} ({len(prompts)} prompts)")
    print(f"{'='*70}")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        comp_display = completion.replace('\n', '\\n')
        if len(comp_display) > 140:
            comp_display = comp_display[:140] + '...'
        print(f"  {prompt}")
        print(f"    -> {comp_display}")
