#[macro_use]
extern crate vulkano;
extern crate image;
extern crate vulkano_shaders;

use std::path::Path;
use libc::c_char;
use std::ffi::CStr;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{Dimensions, ImmutableImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::sync::GpuFuture;

use image::{ImageBuffer, Rgba};

use std::sync::Arc;
use vulkano::device::Features;
use vulkano::instance::InstanceExtensions;
use vulkano::image::StorageImage;

mod quad_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "shaders/quad.vert.glsl"
    }
}

mod expand_known_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        // path: "shaders/test.frag.glsl"
        path: "shaders/expandKnown.frag.glsl"
    }
}

mod gathering_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/gathering.frag.glsl"
    }
}

mod refinement_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/refinement.frag.glsl"
    }
}

mod local_smooth_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/localSmooth.frag.glsl"
    }
}

#[derive(Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

pub struct Shared {
    queue: Arc<vulkano::device::Queue>,
    device: Arc<vulkano::device::Device>,
    size: (u32, u32),
    future: Option<Box<GpuFuture>>,
    ori_state: Arc<ImmutableImage<Format>>,
    tri_extend: Arc<StorageImage<Format>>,
    fb_state: Arc<StorageImage<Format>>,
    back_state: Arc<StorageImage<Format>>,
    fore_state: Arc<StorageImage<Format>>,
    ac_state: Arc<StorageImage<Format>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    buf: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl Shared {
    pub fn new<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        // println!( "Using device: {} (type: {:?})", physical.name(), physical.ty() );
        let queue_family = physical.queue_families().find(|&q| q.supports_graphics()).expect("couldn't find a graphical queue family");
        let (device, mut queues) = {
            Device::new(
                physical,
                &Features::none(),
                &DeviceExtensions::none(),
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        let queue = queues.next().unwrap();
        let vertexs = vec![
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [ 1.0, -1.0] },
            Vertex { position: [-1.0,  1.0] },
            Vertex { position: [ 1.0,  1.0] },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertexs.into_iter()).unwrap();

        let image = image::open(path).unwrap().to_rgba();
        let size = image.dimensions();
        let image = image.into_raw().clone();
        let (width, height) = size;
        let (ori_state, future) = {
            ImmutableImage::from_iter(
                image.iter().cloned(),
                Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                Format::R8G8B8A8Srgb,
                queue.clone(),
            ).unwrap()
        };
        let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0 .. width * height * 4).map(|_| 0u8))
            .expect("failed to create buffer");

        let tri_extend = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let fb_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let back_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let fore_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let ac_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

        Self {
            queue: queue,
            device: device,
            size: size,
            future: Some(Box::new(future) as Box<GpuFuture>),
            ori_state: ori_state,
            tri_extend: tri_extend,
            fb_state: fb_state,
            back_state: back_state,
            fore_state: fore_state,
            ac_state: ac_state,
            buf: buf,
            vertex_buffer: vertex_buffer,
        }
    }
    
    pub fn run<P>(&mut self, tri_path: P, output_path: P)
    where
        P: AsRef<Path>,
    {
        let (width, height) = self.size;
        let (tri_state, future) = {
            let image = image::open(tri_path).unwrap().to_rgba();
            let image = image.into_raw().clone();
            ImmutableImage::from_iter(
                image.iter().cloned(),
                Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                Format::R8G8B8A8Srgb,
                self.queue.clone(),
            )
            .unwrap()
        };
        let f = self.future.take().unwrap();
        self.future = Some(Box::new(f.join(future)) as Box<GpuFuture>);

        self.expand_known(tri_state);
        self.gathering();
        self.refinement();
        self.local_smooth();

        self.future.take().unwrap().then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();
            
        let buffer_content = self.buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, &buffer_content[..]).unwrap();
        image.save(output_path).unwrap();
    }

    fn expand_known(&mut self, tri_state: Arc<ImmutableImage<Format>>) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let image = &self.tri_extend;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap());

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = expand_known_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = expand_known_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            kC: 5.0 / 255.0,
            kI: 10.0,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn gathering(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;

        let image = &self.fb_state;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap());

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = gathering_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = gathering_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            isTest: 0,
            kG: 4,
            searchRange: (width+height) as i32,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn refinement(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;
        let fb_state = &self.fb_state;
        let refine_f = &self.fore_state;
        let refine_b = &self.back_state;
        let refine_ac = &self.ac_state;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                },
                color1: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                },
                color2: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color, color1, color2],
                depth_stencil: {}
            }
        ).unwrap()); 

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(refine_f.clone()).unwrap()
            .add(refine_b.clone()).unwrap()
            .add(refine_ac.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = refinement_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = refinement_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            isTest: 0,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .add_sampled_image(fb_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![
                [0.0, 0.0, 1.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
            ]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn local_smooth(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;
        let back_state = &self.back_state;
        let fore_state = &self.fore_state;
        let ac_state = &self.ac_state;
        let buf = &self.buf;

        let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()); 

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = local_smooth_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = local_smooth_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .add_sampled_image(fore_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(back_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(ac_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }
}

#[no_mangle]
pub extern "C" fn run(path: *const c_char, tri_path: *const c_char, output_path: *const c_char) {
	let path = unsafe {
		assert!(!path.is_null());
		CStr::from_ptr(path)
	};
	let path = path.to_str().unwrap();
	let tri_path = unsafe {
		assert!(!tri_path.is_null());
		CStr::from_ptr(tri_path)
	};
	let tri_path = tri_path.to_str().unwrap();
	let output_path = unsafe {
		assert!(!output_path.is_null());
		CStr::from_ptr(output_path)
	};
	let output_path = output_path.to_str().unwrap();
	let mut shared = Shared::new(path);
	shared.run(tri_path, output_path);
}
