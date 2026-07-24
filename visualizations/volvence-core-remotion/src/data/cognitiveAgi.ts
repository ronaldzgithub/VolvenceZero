import rawDurations from './cognitiveAgiDurations.json';
import rawScenes from './cognitiveAgi.json';

export type CognitiveCaption = {
  from: number;
  to: number;
  text: string;
};

export type CognitiveSceneId =
  | 'loop'
  | 'coverage'
  | 'handoff'
  | 'cognitive'
  | 'relationship'
  | 'system'
  | 'case'
  | 'flywheel'
  | 'team'
  | 'outro';

export type CognitiveScene = {
  id: CognitiveSceneId;
  number: string;
  eyebrow: string;
  title: string;
  narration: string;
  captions: CognitiveCaption[];
  pauseAfter: number;
  audioDuration: number;
  startFrame: number;
  durationInFrames: number;
};

export const COGNITIVE_AGI_FPS = 30;

const durations = rawDurations as Record<CognitiveSceneId, number>;
const sceneInput = rawScenes as Array<
  Omit<CognitiveScene, 'audioDuration' | 'startFrame' | 'durationInFrames'>
>;

let nextStartFrame = 0;

export const cognitiveScenes: CognitiveScene[] = sceneInput.map((scene) => {
  const audioDuration = durations[scene.id];
  if (!Number.isFinite(audioDuration)) {
    throw new Error(`Missing generated audio duration for scene: ${scene.id}`);
  }

  const durationInFrames = Math.ceil(
    (audioDuration + scene.pauseAfter) * COGNITIVE_AGI_FPS,
  );
  const scheduled: CognitiveScene = {
    ...scene,
    audioDuration,
    startFrame: nextStartFrame,
    durationInFrames,
  };
  nextStartFrame += durationInFrames;
  return scheduled;
});

export const COGNITIVE_AGI_DURATION_FRAMES = nextStartFrame;
