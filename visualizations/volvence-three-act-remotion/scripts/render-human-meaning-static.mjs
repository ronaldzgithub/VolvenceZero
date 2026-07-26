import {mkdirSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';

const root = resolve(import.meta.dirname, '..');
const outDir = resolve(root, 'out');
mkdirSync(outDir, {recursive: true});

const segments = [
  ['h7-final-00-opening.png', 5],
  ['h7-final-01-parents-question.png', 18.314],
  ['h7-final-02-parents-advisor.png', 7.369],
  ['h7-final-03-child-growth.png', 21.598],
  ['h7-final-04-child-understood.png', 21.887],
  ['h7-final-05-youth.png', 21.486],
  ['h7-final-06-startup.png', 24.054],
  ['h7-final-07-moonlight.png', 19.821],
  ['h7-final-08-past-self.png', 9.566],
  ['h7-final-09-ancestor.png', 16.563],
  ['h7-final-10-legacy.png', 39.831],
  ['h7-final-11-ending.png', 8],
];

const args = ['-y'];
for (const [file, duration] of segments) {
  args.push(
    '-loop',
    '1',
    '-framerate',
    '30',
    '-t',
    String(duration),
    '-i',
    resolve(outDir, file),
  );
}

const narrationIndex = segments.length;
const heartbeatIndex = narrationIndex + 1;
args.push(
  '-i',
  resolve(
    root,
    'public/human-meaning-standalone/volvence-should-be-remembered-doubao-tts.wav',
  ),
  '-i',
  resolve(root, 'public/human-meaning-standalone/heartbeat.wav'),
);

const videoFilters = segments.map(([, duration], index) => {
  const fadeOut = Math.max(0, duration - 0.35).toFixed(3);
  return [
    `[${index}:v]`,
    'scale=1920:1080:force_original_aspect_ratio=decrease,',
    'pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,',
    'format=yuv420p,',
    `fade=t=in:st=0:d=0.35,fade=t=out:st=${fadeOut}:d=0.35,`,
    'setpts=PTS-STARTPTS',
    `[v${index}]`,
  ].join('');
});

const concatInputs = segments.map((_, index) => `[v${index}]`).join('');
const filter = [
  ...videoFilters,
  `${concatInputs}concat=n=${segments.length}:v=1:a=0[vout]`,
  `[${narrationIndex}:a]adelay=5000:all=1,volume=0.98[narration]`,
  `[${heartbeatIndex}:a]afade=t=in:st=0:d=0.8,afade=t=out:st=12:d=5,volume=0.6[heartbeat]`,
  '[narration][heartbeat]amix=inputs=2:duration=longest:normalize=0[aout]',
].join(';');

args.push(
  '-filter_complex',
  filter,
  '-map',
  '[vout]',
  '-map',
  '[aout]',
  '-r',
  '30',
  '-c:v',
  'libx264',
  '-preset',
  'medium',
  '-crf',
  '18',
  '-c:a',
  'aac',
  '-b:a',
  '192k',
  '-movflags',
  '+faststart',
  resolve(outDir, 'volvence-human-meaning-standalone.mp4'),
);

const result = spawnSync('ffmpeg', args, {
  cwd: root,
  stdio: 'inherit',
});

if (result.error) {
  throw result.error;
}

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
