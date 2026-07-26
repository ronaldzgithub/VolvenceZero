import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import {
  HUMAN_MEANING_AUDIO_FRAMES,
  HUMAN_MEANING_DURATION_FRAMES,
  HUMAN_MEANING_FPS,
  HUMAN_MEANING_INTRO_FRAMES,
  humanMeaningNarration,
  type HumanMeaningNarrationLine,
} from './data/humanMeaningStandalone';
import './human-meaning-standalone.css';

export {
  HUMAN_MEANING_DURATION_FRAMES,
  HUMAN_MEANING_FPS,
};

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const sceneOpacity = (
  frame: number,
  line: HumanMeaningNarrationLine,
): number => {
  const start = HUMAN_MEANING_INTRO_FRAMES + line.start * HUMAN_MEANING_FPS;
  const end = HUMAN_MEANING_INTRO_FRAMES + line.end * HUMAN_MEANING_FPS;
  return interpolate(
    frame,
    [start - 16, start + 22, end - 22, end + 16],
    [0, 1, 1, 0],
    clamp,
  );
};

const Heartbeat: React.FC<{frame: number}> = ({frame}) => {
  const beat = frame % 25;
  const firstPulse = interpolate(beat, [0, 3, 8], [0.72, 1.12, 0.72], clamp);
  const secondPulse = interpolate(beat, [8, 11, 16], [0.72, 1.05, 0.72], clamp);
  const scale = Math.max(firstPulse, secondPulse);
  const line = Array.from({length: 34}, (_, index) => {
    const distance = Math.abs(index - 17);
    const peak = distance === 0 ? -36 : distance === 1 ? 23 : 0;
    const secondary = index === 20 ? -13 : index === 21 ? 9 : 0;
    return 9 + peak + secondary;
  });

  return (
    <div className="h7-heartbeat">
      <div className="h7-heart-orbit" style={{transform: `scale(${scale})`}}>
        <div className="h7-heart-dot" />
      </div>
      <svg viewBox="0 0 510 70" aria-hidden="true">
        <polyline
          points={line.map((value, index) => `${index * 15},${value + 30}`).join(' ')}
        />
      </svg>
    </div>
  );
};

const CinematicScene: React.FC<{
  frame: number;
  line: HumanMeaningNarrationLine;
}> = ({frame, line}) => {
  const start = HUMAN_MEANING_INTRO_FRAMES + line.start * HUMAN_MEANING_FPS;
  const end = HUMAN_MEANING_INTRO_FRAMES + line.end * HUMAN_MEANING_FPS;
  const localProgress = interpolate(frame, [start, end], [0, 1], clamp);
  const opacity = sceneOpacity(frame, line);
  const copyY = interpolate(frame, [start, start + 34], [24, 0], clamp);

  return (
    <AbsoluteFill className={`h7-scene h7-align-${line.align}`} style={{opacity}}>
      <Img
        className="h7-scene-image"
        src={staticFile(`human-meaning-standalone/scenes/${line.image}`)}
        style={{transform: `scale(${1.025 + localProgress * 0.045})`}}
      />
      <div className="h7-scene-grade" />
      <div className="h7-scene-vignette" />
      <div
        className="h7-scene-copy"
        style={{
          opacity: interpolate(frame, [start + 6, start + 32], [0, 1], clamp),
          transform: `translateY(${copyY}px)`,
        }}
      >
        <span>{line.chapter}</span>
        <strong>{line.headline}</strong>
        <i />
        <b>{line.supporting}</b>
      </div>
      <div className="h7-life-marker">
        <span>{String(line.sequence).padStart(2, '0')}</span>
        <i />
        <b>一生</b>
      </div>
    </AbsoluteFill>
  );
};

export const VolvenceHumanMeaningStandalone: React.FC = () => {
  const frame = useCurrentFrame();
  const introFade = interpolate(frame, [0, 28, 112, 150], [0, 1, 1, 0], clamp);
  const titleLift = interpolate(frame, [26, 76], [28, 0], clamp);
  const audioEnd = HUMAN_MEANING_INTRO_FRAMES + HUMAN_MEANING_AUDIO_FRAMES;
  const outroOpacity = interpolate(frame, [audioEnd - 4, audioEnd + 38], [0, 1], clamp);
  const finalGlow = interpolate(frame, [audioEnd + 20, audioEnd + 150], [0.15, 0.72], clamp);

  return (
    <AbsoluteFill className="h7-film">
      <Audio
        src={staticFile('human-meaning-standalone/heartbeat.wav')}
        volume={(audioFrame) =>
          interpolate(audioFrame, [0, 30, 360, 510], [0, 0.7, 0.45, 0], clamp)
        }
      />

      {humanMeaningNarration.map((line) => (
        <CinematicScene key={line.sequence} frame={frame} line={line} />
      ))}

      <AbsoluteFill className="h7-opening" style={{opacity: introFade}}>
        <Heartbeat frame={frame} />
        <div
          className="h7-opening-title"
          style={{transform: `translateY(${titleLift}px)`}}
        >
          <span>VOLVENCE · 第七幕</span>
          <strong>《应该被记住》</strong>
        </div>
      </AbsoluteFill>

      <div className="h7-film-grain" />

      <AbsoluteFill className="h7-ending" style={{opacity: outroOpacity}}>
        <div className="h7-ending-glow" style={{opacity: finalGlow}} />
        <div className="h7-ending-copy">
          <span>VOLVENCE</span>
          <strong>模型进化，生命涌现。</strong>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
