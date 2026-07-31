import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("truthful evidence surface", () => {
  it("starts in BLOCK and never presents replay as passed evidence", () => {
    render(<App />);
    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(
      screen.getByText(/尚无通过冻结门槛的正式 artifact/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "创建真实闭环" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "黄油" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "燃烧火柴" }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "暂停" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "停止" })).toBeDisabled();
    expect(screen.getByLabelText("遥测对象")).toBeDisabled();
    expect(
      screen.getByText(/未加载（冷启动在线学习）/),
    ).toBeInTheDocument();
  });
});
