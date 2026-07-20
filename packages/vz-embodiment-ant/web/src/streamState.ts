import type {
  AppFrame,
  DisturbanceRecord,
  RunStatus,
  StreamState,
} from "./types";

export const initialStreamState: StreamState = {
  lastSequence: 0,
  frame: null,
  status: null,
  disturbances: [],
  connectionError: "",
};

export type StreamAction =
  | { type: "reset" }
  | { type: "frame"; sequence: number; payload: AppFrame }
  | { type: "status"; sequence: number; payload: RunStatus }
  | {
      type: "disturbance";
      sequence: number;
      payload: DisturbanceRecord;
    }
  | { type: "connection"; message: string };

export function reduceStreamState(
  state: StreamState,
  action: StreamAction,
): StreamState {
  if (action.type === "reset") return initialStreamState;
  if (action.type === "connection") {
    return { ...state, connectionError: action.message };
  }
  if (action.sequence <= state.lastSequence) return state;
  if (action.type === "frame") {
    return {
      ...state,
      lastSequence: action.sequence,
      frame: action.payload,
    };
  }
  if (action.type === "status") {
    return {
      ...state,
      lastSequence: action.sequence,
      status: action.payload,
    };
  }
  return {
    ...state,
    lastSequence: action.sequence,
    disturbances: [...state.disturbances.slice(-19), action.payload],
  };
}
