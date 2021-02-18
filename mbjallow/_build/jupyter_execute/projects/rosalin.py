# Rosalin Bioinformatics Challenge
---


#### About Rosalin
Rosalin is platform created to support bioinformatic enthusiast to learn through problem solving. It is a project inspired by [Project Euler](https://projecteuler.net), [Google Code Jam](https://codingcompetitions.withgoogle.com/codejam). [more about rosalin](http://rosalind.info/about/)

import random 
import collections
# validate DNA sequence helper function

Nucleotides = ['A','C','G','T']

def val_DNA_Seq(dna_string):
  seq2upper = dna_string.upper()
  for seq in seq2upper:
    if seq not in Nucleotides:
      return false
  return seq2upper


# creating a random  DNA sequence to test the validate helper function


rand_DNA_String = ''.join([random.choice(Nucleotides) for seg in range(10)])

sample_DNA_String = val_DNA_Seq(rand_DNA_String)
sample_DNA_String


# counting necleotide frequencies 

def count_Nuc_Frequency(dna_string):
  nuc_string_counter = {'A':0, 'C':0, 'G':0, 'T':0}
  for seq in dna_string:
    nuc_string_counter[seq] += 1
  return nuc_string_counter


#  using the in built collection method

def count_Nuc_Frequency_v2(dna_string):
  return dict(collections.Counter(dna_string))

# test the function using the random generated dna sequence
print(count_Nuc_Frequency_v2(sample_DNA_String))
print(count_Nuc_Frequency(sample_DNA_String))

#  rosalind challege counting nucleotide frequency
dna_string = 'AGGGAACTTATGTAGGTGGCATTTAGACGGACCTCATAAAAAGCTTGGCCAGATACAGTGCTGAAGCGTCGCGGGTAGGTCCGGACCGTACTTTGTAATGAGCTGCGGGCCTCGGACTGCAATCGCCTTATCCCTTTGCGTGGATTACGCGGCTAGCTGCATTACTGTCGGGTACGGTCACCAGGGATACGAAGGGTAATATAAGATAGCTGAGAGCCTATTATGGAGGCAGAGCTATGATCACCGAAACATACTCTTCCATGGCTATCCAATCGATTAGGTTAGGCACTATGAGTTCTGACTAAACATGTTATTGACGTCAGTCCCAGGCGCCACCATAGCCGTGAGAGAGTTTAGCACGTTCGCTTCCAGGACTTACTTGGTCATAGCCGCTGGGAACCCGCGATATCATTCGGCCAGGCGTCCCCCACCAGAAGCCACGCAACGGGACCGAATCCTCACGGCAGGTTACTAGTTGTAGAGGCTAGCTCTTAAGAGGCAGAGCACGCAAGCGTTTCGTTTTATGCCCGATAAGCTTTTAGATGTTACAGACTAAAGAGCCCTGTTGGATGCATAGAGCACTTACTTTAAACTCATGGAAAAAAGCTTTTAATTATTGGCGTAGATATCGCGTGAAATGCCTGGTATCCGTAGGGGCTGGCCATCTTGACACCTCGAGCTGCCTACATGTGCCAGGAGATACCTGGTAATTCGTTACTCGATTCGCGACGGACCTATGTATTCACGCAGTCGACTTCGCTCCTGCACTTTGAACTAATCTGATAGGCAGAATCACTGGGTGTTATTTTCATGCGTATCGGTTACGTCCACGTCCATGTTAAAACGAAAACAGTATGCGGGCTAAGGCATTGTCGGTTACCCACCCGCTGCGTGGATC'


nuc_counter = count_Nuc_Frequency(dna_string)
print(' '.join([str(val) for key, val in nuc_counter.items()]))

# rosalind python village challenge
# problem 1 install python and import this module

import this

# print the hypotenus of right triangle

def hypotenus(a,b):
  print("{}".format(a**2 + b**2))

hypotenus(810 ,903)

def sliceText(txt, a,b,c,d):
  

  print("{} {}".format(txt[a:b+1], txt[c:d+1]))



a,b,c,d = 51,58,151,158
text = 'jOweVhwIhndHjF92sKbEvBq8bi0YUueb1CcxfcsWTDaMLpmF8qpAmphiumaNUQokYgIZ68CGPTiMxiG92v6yfogAy337zzPhg2x2o6rWGi2fDER24SBvwvJLDy2YvgySfQRXVaQ2dOTpdiHBw0zVW6PalbigulaxXIQinUbesd'
sliceText(text,a,b,c,d)


def sumOdds(a,b):
  if a>b or b>10000:
    return 0
  sumodds = 0
  for i in range(a,b+1):
    if i%2 != 0:
      sumodds = sumodds+i
  return sumodds
  
a, b = 4174, 8672
sumOdds(a,b)

# reading and writing on files 
def read_write(in_path, out_path):
  with open(in_path,'r') as rf:
    text = [txt for  num, txt in  enumerate(rf.readlines())
    if num % 2 != 0]

  with open(out_path,'w') as wf:
    wf.write(''.join([newtext for newtext in text]))
  return text


read_write('/content/rosalind_ini5.txt', '/content/out.txt')

# count words
def count_words(text):
  words_counter = {}
  for word in text.split(' '):
    if word in words_counter:
      words_counter[word] += 1
    else:
      words_counter[word] = 1
  
  for key, value in words_counter.items():
    print(key, value) 
    

txt = 'When I find myself in times of trouble Mother Mary comes to me Speaking words of wisdom let it be And in my hour of darkness she is standing right in front of me Speaking words of wisdom let it be Let it be let it be let it be let it be Whisper words of wisdom let it be And when the broken hearted people living in the world agree There will be an answer let it be For though they may be parted there is still a chance that they will see There will be an answer let it be Let it be let it be let it be let it be There will be an answer let it be Let it be let it be let it be let it be Whisper words of wisdom let it be Let it be let it be let it be let it be Whisper words of wisdom let it be And when the night is cloudy there is still a light that shines on me Shine until tomorrow let it be I wake up to the sound of music Mother Mary comes to me Speaking words of wisdom let it be Let it be let it be let it be yeah let it be There will be an answer let it be Let it be let it be let it be yeah let it be Whisper words of wisdom let it be'
count_words(txt)