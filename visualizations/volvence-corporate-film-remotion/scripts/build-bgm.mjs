import {mkdirSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';

const root = resolve(import.meta.dirname, '..');
const outputDir = resolve(root, 'public/music');
const output = resolve(outputDir, 'volvence-corporate-bed.wav');
mkdirSync(outputDir, {recursive: true});

// Original procedural score: sparse plucked harmonics gradually give way to
// a warmer sustained pad. It stays intentionally restrained under narration.
const expression = [
  '0.028*(',
  'sin(2*PI*110*t)',
  '+0.62*sin(2*PI*164.81*t)',
  '+0.46*sin(2*PI*220*t)',
  ')*(0.28+0.72*exp(-2.8*mod(t\\,2.75)))',
  '+0.012*(min(1\\,max(0\\,(t-24)/46)))*(',
  'sin(2*PI*55*t)',
  '+0.55*sin(2*PI*82.41*t)',
  '+0.38*sin(2*PI*110*t)',
  ')',
  '+0.006*(min(1\\,max(0\\,(t-55)/35)))*',
  'sin(2*PI*329.63*t)*(0.65+0.35*sin(2*PI*0.17*t))',
].join('');

const result = spawnSync(
  'ffmpeg',
  [
    '-y',
    '-v',
    'error',
    '-f',
    'lavfi',
    '-i',
    `aevalsrc=${expression}:s=48000:d=110`,
    '-af',
    [
      'highpass=f=42',
      'lowpass=f=4200',
      'aecho=0.8:0.72:260|520:0.14|0.08',
      'afade=t=in:st=0:d=2.5',
      'afade=t=out:st=106:d=4',
      'loudnorm=I=-27:TP=-5:LRA=5',
      'aformat=sample_fmts=fltp:channel_layouts=stereo',
    ].join(','),
    '-ar',
    '48000',
    '-ac',
    '2',
    '-c:a',
    'pcm_s16le',
    output,
  ],
  {cwd: root, stdio: 'inherit'},
);

if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
console.log(`Music bed: ${output}`);
