#! /usr/bin/perl
use strict;
my $file_name=$ARGV[0];
my $th=$ARGV[1];
my $h=$ARGV[2];
my $bench=$file_name;
$bench=~s/_K.*$//g;
my $key_size=$file_name;
$key_size=~s/^.*_K//g;
$key_size=~s/_DMUX.*$//g;
my $correct_key="";
my %connect_from_all=();
my %connect_from=();
my %connect_to=();
my %seen_correct=();
my %connect_from_correct=();
my %connect_to_correct=();
my %connect_from_correct_val=();
my %connect_from_val=();
my %cells=();

my %dec_key_bits=();
my %key_bits=();
open (FH,'<', "./data/${file_name}/cell.txt") or die $!;

while (<FH>){
my $line= $_;
my @columns=split(/\s+/,$line);
$cells{$columns[4]}=$columns[0];
}
close(FH);

open (FH,'<', "./data/${file_name}/links_test_${h}__pred.txt") or die $!;
while (<FH>){
my $line= $_;
my @columns=split(/\s+/,$line);

$connect_from_all{$columns[0]." ".$columns[1]}= $columns[2];
push @{$connect_from_correct{$columns[0]}}, $columns[0]." ".$columns[1];
push @{$connect_from_correct_val{$columns[0]}}, $columns[2];
push @{$connect_to_correct{$columns[1]}}, $columns[0]." ".$columns[1];
}
close(FH);
open (FH,'<', "./data/${file_name}/link_test_n_${h}__pred.txt") or die $!;
while (<FH>){
my $line= $_;
my @columns=split(/\s+/,$line);
 $connect_from_all{$columns[0]." ".$columns[1]}= $columns[2];
push @{$connect_from{$columns[0]}}, $columns[0]." ".$columns[1];
push @{$connect_from_val{$columns[0]}}, $columns[2];
push @{$connect_to{$columns[1]}}, $columns[0]." ".$columns[1];
}
close(FH);



open (FH,'<', "./data/${file_name}/${bench}_K${key_size}_v1.bench") or die $!;
while (<FH>){
my $line= $_;
if ($line=~/#key=(\d+)\s*/){
$correct_key=$1;
print "Correct key is $correct_key\n";
}
if ($line=~m/keyinput/ && !($line=~m/INPUT/)){
chomp $line;
if ($line=~/^\s*(\S+)_from_mux\s*\=\s*MUX\(keyinput(\d+)\s*\,\s*(\S+)\s*\,\s*(\S+)\s*\)$/){
my $output=$1;
my $key_bit=$2;
my $path0=$3;
my $path1=$4;
push @{$key_bits{$key_bit}}, "$output $path0 $path1";

}


}
}
close FH;
my %to_check_key_bits=%key_bits;
foreach my $key (keys %key_bits) {
if(exists($to_check_key_bits{$key})){
delete($to_check_key_bits{$key});
}
else {
next;
}
my @connections=@{$key_bits{$key}};
my $length = @connections;
my $line=$connections[0];
my @columns=split(/\s+/,$line);
my $output=$columns[0];
my $path0=$columns[1];
my $path1=$columns[2];
my $lik0= $connect_from_all{$cells{$path0}." ".$cells{$output}};
my $lik1= $connect_from_all{$cells{$path1}." ".$cells{$output}};
my $dif=abs($lik0 - $lik1);

my $flag=0;
if ($length==1){
foreach my $key_two (keys %to_check_key_bits) {
my @connections_two=@{$to_check_key_bits{$key_two}};
my $line_two=$connections_two[0];
my @columns_two=split(/\s+/,$line_two);
my $output_two=$columns_two[0];
my $path0_two=$columns_two[1];
my $path1_two=$columns_two[2];
if (($path0_two eq $path0) && ($path1_two eq $path1)){
$flag=1;
delete($to_check_key_bits{$key_two});
my $lik0_two= $connect_from_all{$cells{$path0}." ".$cells{$output_two}};
my $lik1_two= $connect_from_all{$cells{$path1}." ".$cells{$output_two}};
my $dif2=abs($lik0_two - $lik1_two);
if ($dif>=$th || $dif2>=$th){
if ($dif>$dif2){
if ($lik0>$lik1)
{$dec_key_bits{$key}="0";
$dec_key_bits{$key_two}="1";}
else{$dec_key_bits{$key}="1";
$dec_key_bits{$key_two}="0";}}#end if difference 1 is greater than dif2
elsif ($dif2>$dif){
if ($lik0_two>$lik1_two)
{$dec_key_bits{$key_two}="0";
$dec_key_bits{$key}="1";}
else{$dec_key_bits{$key_two}="1";
$dec_key_bits{$key}="0";}}#end if difference 2 is greater than dif2
else { #they are equal
if ($lik0> $lik1){if ($lik0>$lik0_two){
$dec_key_bits{$key}="0";
$dec_key_bits{$key_two}="1";}
}
elsif ($lik1> $lik0){if ($lik1>$lik1_two){
$dec_key_bits{$key_two}="0";
$dec_key_bits{$key}="1";}
}
else{
$dec_key_bits{$key}="X";
$dec_key_bits{$key_two}="X";
}}
}#checking both thresholds
else {
$dec_key_bits{$key}="X";
$dec_key_bits{$key_two}="X";
}next;}}
if ($flag==0){
if ($dif>=$th){
if ($lik0>$lik1)
{$dec_key_bits{$key}="0";}
else{$dec_key_bits{$key}="1";}}
else {$dec_key_bits{$key}="X";}
}
}
else {
my $line_two=$connections[1];
my @columns_two=split(/\s+/,$line_two);
my $output_two=$columns_two[0];
my $lik0_two= $connect_from_all{$cells{$path0}." ".$cells{$output_two}};
my $lik1_two= $connect_from_all{$cells{$path1}." ".$cells{$output_two}};
my $dif2=abs($lik0_two - $lik1_two);
if ($dif>=$th || $dif2>=$th){
if ($dif>$dif2){
if ($lik0>$lik1)
{
$dec_key_bits{$key}="0";}
else{
$dec_key_bits{$key}="1";}}#end if difference 1 is greater than dif2
elsif ($dif2>$dif){
if ($lik0_two>$lik1_two)
{
$dec_key_bits{$key}="1";}
else{
$dec_key_bits{$key}="0";}}#end if difference 2 is greater than dif2
else { #they are equal
if ($lik0> $lik1){if ($lik0>$lik0_two){
$dec_key_bits{$key}="0";}
}
elsif ($lik1> $lik0){if ($lik1>$lik1_two){
$dec_key_bits{$key}="1";}
}
else{
$dec_key_bits{$key}="X";
}}
}#checking both thresholds
else {
$dec_key_bits{$key}="X";
}
}
}
my $i=0;
my $un=0;
my $w=0;
my $c=0;
my $report="";
print "KEY=";
while ($i< $key_size){
my $true = substr($correct_key, $i, 1);
my $pred=$dec_key_bits{$i};
print "$pred";
if ($true eq $pred){$c++;}
else {
if ($pred eq "X"){
$un++;
}
else{
$report .="keybit$i is wrong\n";
$w++;
}

}
$i++;
}
print "\n";
my $accuracy=$c/$key_size;
my $prec= ($c+$un)/$key_size;
my $kpa= $c/($key_size-$un);
print "Correct key-bits are $c, wrong key-bits are $w, and tie are $un\n";
print "$accuracy $prec $kpa\n";
print "Acc. $accuracy, Prec. $prec, KPA $kpa\n";
print "$report\n";
