using UnityEngine;

public class FrameByFrameAnimation : MonoBehaviour
{
    public Animation animation;
    public string animationName;
    private float frameTime = 0.0f; // The current frame time

    void Start()
    {
        animation[animationName].speed = 0; // Stop automatic playback
    }

    void Update()
    {
        frameTime += Time.deltaTime; // Increase the frame time by the time since the last frame
        animation[animationName].normalizedTime = frameTime / animation[animationName].length; // Set the current time of the animation
        animation.Sample(); // Update the animation
    }
}