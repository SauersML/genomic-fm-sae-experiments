BEGIN{ FS="\t"; OFS="\t" }
/^#/ { next }
{
  ref=$4; alt=$5;
  # skip symbolic/multi-allelic leftovers
  if (alt ~ /[<>\[\]]/) next;
  # split on comma should not occur (norm -m -any) but guard
  if (alt ~ /,/) next;
  rl=length(ref); al=length(alt);
  d = al - rl; if (d<0) d=-d;
  if (d >= 50) {
    print $1,$2,$3,ref,alt,$8;
  }
}
