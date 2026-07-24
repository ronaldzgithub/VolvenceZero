import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const projectRoot = path.resolve(import.meta.dirname, '..');
const scenesPath = path.join(projectRoot, 'src/data/cognitiveAgi.json');
const durationsPath = path.join(
  projectRoot,
  'src/data/cognitiveAgiDurations.json',
);
const audioDir = path.join(projectRoot, 'public/cognitive-agi');
const scenes = JSON.parse(fs.readFileSync(scenesPath, 'utf8'));

fs.mkdirSync(audioDir, {recursive: true});

const run = (command, args) => {
  const result = spawnSync(command, args, {encoding: 'utf8'});
  if (result.status !== 0) {
    throw new Error(
      `${command} failed (${result.status}): ${result.stderr || result.stdout}`,
    );
  }
  return result.stdout.trim();
};

const durations = {};

for (const scene of scenes) {
  const aiffPath = path.join(audioDir, `${scene.id}.aiff`);
  const audioPath = path.join(audioDir, `${scene.id}.m4a`);

  run('say', [
    '-v',
    'Tingting',
    '-r',
    '208',
    '-o',
    aiffPath,
    scene.narration,
  ]);
  run('ffmpeg', [
    '-y',
    '-loglevel',
    'error',
    '-i',
    aiffPath,
    '-ar',
    '44100',
    '-ac',
    '1',
    '-c:a',
    'aac',
    '-b:a',
    '128k',
    audioPath,
  ]);
  fs.unlinkSync(aiffPath);

  durations[scene.id] = Number(
    run('ffprobe', [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      audioPath,
    ]),
  );
}

fs.writeFileSync(durationsPath, `${JSON.stringify(durations, null, 2)}\n`);
console.log(JSON.stringify(durations, null, 2));
